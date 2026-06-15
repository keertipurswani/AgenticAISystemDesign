import ast
import textwrap


# ---------------------------------------------------------------------------
# Helper — pretty-print an AST node tree with indentation
# ---------------------------------------------------------------------------
def show_tree(code: str) -> None:


   print("\n--- SOURCE ---")
   print(code.strip())


   print("\n--- AST dump ---")
   tree = ast.parse(code)
   print(ast.dump(tree, indent=2))


# ---------------------------------------------------------------------------
# EXAMPLE 1  — a simple expression
# ---------------------------------------------------------------------------


# Key observations Python produces for `x = 1 + 2`:
#
#   Module
#   └── Assign
#       ├── targets: [Name(id='x')]
#       └── value:   BinOp
#                    ├── left:  Constant(value=1)
#                    ├── op:    Add()
#                    └── right: Constant(value=2)


show_tree("x = 1 + 2")




# ---------------------------------------------------------------------------
# EXAMPLE 2  — a function definition
# ---------------------------------------------------------------------------


# The tree shows:
#   FunctionDef
#   ├── name: 'greet'
#   ├── args: arguments  (name, annotation=str)
#   ├── returns: Name(id='str')
#   └── body:
#       └── Return
#           └── BinOp(Constant("Hello, ") + Name(id='name'))


show_tree(textwrap.dedent("""\
   def greet(name: str) -> str:
       return "Hello, " + name
"""))




# ---------------------------------------------------------------------------
# EXAMPLE 3  — a class with methods
# ---------------------------------------------------------------------------


show_tree(textwrap.dedent("""\
   class Dog:
       def __init__(self, name):
           self.name = name


       def bark(self):
           return f"Woof! I am {self.name}"
"""))
