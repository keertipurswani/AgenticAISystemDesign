from pathlib import Path
from dataclasses import dataclass


from tree_sitter import Language, Parser
from tree_sitter_languages import get_language, get_parser


from educosys_claude.observability.logger import get_logger


logger = get_logger(__name__)


# Maps file extension → tree-sitter language name.
# tree-sitter-languages bundles all these grammars, no extra installs needed.
EXTENSION_TO_LANGUAGE = {
  ".py": "python",
  ".js": "javascript",
  ".jsx": "javascript",
  ".ts": "typescript",
  ".tsx": "tsx",
  ".java": "java",
  ".go": "go",
  ".rs": "rust",
  ".cpp": "cpp",
  ".c": "c",
  ".cs": "c_sharp",
  ".rb": "ruby",
  ".php": "php",
  ".swift": "swift",
  ".kt": "kotlin",
  ".sh": "bash",
}




# Text/config files have no meaningful AST — we chunk them by line count instead.
TEXT_EXTENSIONS = {".md", ".txt", ".yaml", ".yml", ".json", ".toml"}
ALL_EXTENSIONS = set(EXTENSION_TO_LANGUAGE.keys()) | TEXT_EXTENSIONS


# Sliding window settings for text files and AST fallback.
# CHUNK_OVERLAP ensures context isn't lost at chunk boundaries.
CHUNK_SIZE = 50
CHUNK_OVERLAP = 10


# Tree-sitter node type names that correspond to a named, indexable block.
# These are consistent across languages — tree-sitter uses the same names where possible.
BLOCK_NODE_TYPES = {
  "function_definition", "function_declaration", "method_definition",
  "arrow_function", "class_definition", "class_declaration",
  "method_declaration", "constructor_declaration", "interface_declaration",
  "function_item",    # Rust uses this instead of function_definition
  "func_declaration", # Go
}


@dataclass
class ParsedChunk:
  name: str
  type: str        # "function" | "class" | "block"
  content: str
  source: str      # absolute path to the file
  start_line: int
  end_line: int


def parse_file(filepath: str) -> list[ParsedChunk]:
  """Entry point — routes to AST parsing or sliding window based on file type."""
  ext = Path(filepath).suffix.lower()


  if ext in TEXT_EXTENSIONS:
      lines = Path(filepath).read_text(encoding="utf-8", errors="ignore").splitlines()
      return _sliding_window(lines, filepath)


  language_name = EXTENSION_TO_LANGUAGE.get(ext)
  if not language_name:
      raise ValueError(f"Unsupported file type: {ext}")


  source = Path(filepath).read_text(encoding="utf-8", errors="ignore")
  return _parse_with_treesitter(source, filepath, language_name)


def _parse_with_treesitter(source: str, filepath: str, language_name: str) -> list[ParsedChunk]:
  """Parse source with the appropriate tree-sitter grammar and extract named blocks."""
  logger.info(f"Parsing {language_name} file: {filepath}")
  parser = get_parser(language_name)
  tree = parser.parse(source.encode())
  lines = source.splitlines()


  chunks = []
  _walk(tree.root_node, source, filepath, chunks, depth=0)


  # If the AST yielded nothing (e.g. a file with only imports), fall back to line chunks
  if not chunks:
      logger.warning(f"No AST blocks found in {filepath}, falling back to sliding window")
      return _sliding_window(lines, filepath)


  logger.info(f"Parsed {len(chunks)} chunks from {filepath}")
  return chunks


def _walk(node, source: str, filepath: str, chunks: list, depth: int):
  """
  Recursively walk the AST. When a named block node is found, record it and stop
  descending — this keeps chunks at the top level and avoids duplicating nested functions.
  """
  if node.type in BLOCK_NODE_TYPES:
      name = _extract_name(node, source)
      content = source[node.start_byte:node.end_byte]
      chunk_type = "class" if "class" in node.type else "function"
      chunks.append(ParsedChunk(
          name=name,
          type=chunk_type,
          content=content,
          source=filepath,
          start_line=node.start_point[0] + 1,
          end_line=node.end_point[0] + 1,
      ))
      logger.debug(f"  Found {chunk_type} '{name}' (lines {node.start_point[0]+1}-{node.end_point[0]+1})")
      return  # stop here — don't index nested functions/classes as separate chunks


  for child in node.children:
      _walk(child, source, filepath, chunks, depth + 1)


def _extract_name(node, source: str) -> str:
  """Find the identifier child of a block node and return its text as the chunk name."""
  for child in node.children:
      if child.type in ("identifier", "name", "property_identifier"):
          return source[child.start_byte:child.end_byte]
  return node.type  # fallback to node type if no name found


def _sliding_window(lines: list[str], filepath: str) -> list[ParsedChunk]:
  """
  Split lines into overlapping fixed-size chunks.
  Used for text files and as a fallback when AST parsing finds nothing.
  """
  if not lines:
      raise ValueError(f"Empty file: {filepath}")


  chunks = []
  step = CHUNK_SIZE - CHUNK_OVERLAP
  for i, start in enumerate(range(0, len(lines), step)):
      end = min(start + CHUNK_SIZE, len(lines))
      text = "\n".join(lines[start:end]).strip()
      if text:
          chunks.append(ParsedChunk(
              name=f"chunk_{i}", type="block", content=text,
              source=filepath, start_line=start + 1, end_line=end,
          ))
      if end == len(lines):
          break


  logger.info(f"Parsed {len(chunks)} chunks from {filepath}")
  return chunks


def get_source_files(repo_path: str, skip_dirs: list[str] = None) -> list[str]:
  """Recursively find all indexable source files under repo_path, skipping build/env dirs."""
  skip = set(skip_dirs or [".venv", "venv", "__pycache__", ".git", "node_modules", "dist", "build"])
  files = [
      str(path) for path in Path(repo_path).rglob("*")
      if path.suffix.lower() in ALL_EXTENSIONS
      and not any(part in skip for part in path.parts)
  ]
  logger.info(f"Found {len(files)} source files in {repo_path}")
  return files




