# MyAI
MyAI is a Python-based project that combines my module MatVecLib for linear algebra  , and a basic neural network model built with TensorFlow.
Features
    Matrix and Vector Operations:
        Addition, subtraction, and multiplication of matrices
        Determinant and inverse calculations for matrices
        Cross product and dot product for vectors
        Random matrix generation
    Basic AI Model:
        A simple neural network implemented with TensorFlow's Keras API
        Designed to classify user input commands for mathematical operations
    Command Execution:
        The project includes an interactive console-based interface that accepts user commands to perform specific operations

Project Structure
    Matrix: Defines a Matrix class that handles various matrix operations.
    Vector: Defines a Vector class for vector operations like dot product and cross product.
    MyAI: A TensorFlow-based neural network model to classify commands.
    execute_command: A function that interprets and executes commands from the user.

Example Commands

Here are some commands you can use:
    Matrix Operations:
        "add": Adds two matrices.
        "subtract": Subtracts the second matrix from the first.
        "multiply": Multiplies two matrices.
        "determinant": Calculates the determinant of a matrix.
        "invert": Inverts a matrix.
    Vector Operations:
        "dot product": Computes the dot product of two vectors.
        "cross product": Computes the cross product of two vectors.
    AI Commands:
        "Hello": Generates a random matrix and demonstrates basic operations.
