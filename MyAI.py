import tensorflow as tf
class Matrix:
    def __init__(self, matrix=None):
        """
        Initialize a matrix object
        :param matrix: A list representing the matrix
        :return: None
        """
        if matrix is None:
            self.rows = int(input("Enter number of rows: "))
            self.cols = int(input("Enter number of columns: "))
            self.matrix = self._get_matrix_from_input()
        else:
            self.matrix = matrix
            self.rows = len(matrix)
            self.cols = len(matrix[0])

    def __sub__(self, other):
        """
        Subtract two matrices together
        :param other: other matrix
        :return: a new matrix with the subtracted result
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions for subtraction.")
        result_matrix = [[self.matrix[i][j] - other.matrix[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(result_matrix)

    def invert_matrix(self):
        """
        Invert the matrix
        :return: inverted matrix
        """
        if self.rows != self.cols:
            raise ValueError("Matrix must be square to invert.")
        n = self.rows
        I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        copy = [row[:] for row in self.matrix]
        for f in range(n):
            if copy[f][f] == 0:
                raise ValueError("Matrix is not invertible.")
            fS = copy[f][f]
            for i in range(n):
                copy[f][i] /= fS
                I[f][i] /= fS
            for i in range(n):
                if f != i:
                    fs2 = copy[i][f]
                    for j in range(n):
                        copy[i][j] -= fs2 * copy[f][j]
                        I[i][j] -= fs2 * I[f][j]
        return Matrix([[I[j][i] for i in range(n)] for j in range(n)])

    def _get_matrix_from_input(self):
        """
        get a matrix from input
        :return: matrix from input
        """
        matrix = []
        for i in range(self.rows):
            while True:
                try:
                    row = list(map(int, input(f"Enter row {i + 1}: ").split()))
                    if len(row) != self.cols:
                        raise ValueError('Row length does not match the number of columns.')
                    if any(not (0 <= x <= 5) for x in row):
                        raise ValueError('Matrix values must be between 0 and 5.')
                    matrix.append(row)
                    break
                except ValueError as e:
                    print(e)
        return matrix

    def  divinde(self,B):
        """
        Divide two matrices
        :param B : second matrix
        :return: matrix with divide result
        """
        return(Matrix.multiply_matrix(self, Matrix.invert_matrix(B)))

    def random_matrix(self,min_size=1, max_size=100, min_value=0, max_value=10):
        import random
        rows = random.randint(min_size, max_size)
        cols = random.randint(min_size, max_size)
        matrix = [[random.randint(min_value, max_value) for _ in range(cols)] for _ in range(rows)]
        return Matrix(matrix)

    def invert_matrix(self):
        """
        Invert the matrix
        :return:inverted matrix
        """
        if self.rows != self.cols:
            raise ValueError(f'error')
        n = self.rows
        I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        copy = [row[:] for row in self.matrix]
        for f in range(n):
            if copy[f][f] == 0:
                return "Matrix is not invertible"
            fS = copy[f][f]
            for i in range(n):
                copy[f][i] /= fS
                I[f][i] /= fS
            for i in range(n):
                if f != i:
                    fs2 = copy[i][f]
                    for j in range(n):
                        copy[i][j] -= fs2 * copy[f][j]
                        I[i][j] -= fs2 * I[f][j]
        return Matrix([[I[j][i] for i in range(n)] for j in range(n)])
    def determinant(self):
        """
         find the determinant of a matrix
        :return:determinant of matrix
        """
        if self.rows != self.cols:
            raise ValueError(f'error')
        # det = 0
        if self.rows == 2:
            return self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0]
        det = 0
        for i in range(self.rows):
            minor = self._minor(0, i)
            det += ((-1) ** i) * self.matrix[0][i] * minor.determinant()
        return det

    def _minor(self, row, col):
         """
          a function to find the minor of a matrix
          param row: rows of the matrix
          param col: columns of the matrix
         :return:minor matrix
         """
         minor_matrix = [row[:] for row in self.matrix]
         minor_matrix.pop(row)
         for i in range(len(minor_matrix)):
             minor_matrix[i].pop(col)
         return Matrix(minor_matrix)

    def trace(self):
        """
        Calculate the trace of the matrix
        :return: the trace of the matrix
        """
        if self.rows != self.cols:
            raise ValueError("Matrix must be square to calculate trace")
        return sum(self.matrix[i][i] for i in range(self.rows))


    def add_matrix(self,B): #+
        """
         Add two matrices together
        :paramB: second matrix
        :return:a new matrix with the added result
        """
        if self.rows != B.rows or self.cols != B.cols:
            raise ValueError(f'error')
        result = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                result[i][j] = self.matrix[i][j] + B.matrix[i][j]
        return result


    def __mul__(self, other):
        return Matrix(self.multiply_matrix(other))

    def multiply_matrix(self, B):  # *
        if self.cols != B.rows:
            raise ValueError("Cannot multiply: incompatible dimensions.")
        result = [[0 for _ in range(B.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(B.cols):
                for k in range(self.cols):
                    result[i][j] += self.matrix[i][k] * B.matrix[k][j]

        return result

    def transpose(self):
        """
        transpose the matrix
        :return:transposed matrix
        """

        if self.rows != self.cols:
            raise ValueError(f'error')
        new_matrix = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                new_matrix[i][j] = self.matrix[j][i]
        return Matrix(new_matrix)

    def  square(self,another):  #degree == 2
        """
         make matrix in second degree
        :param another:another matrix
        :return: squared matrix
        """
        if self.rows != another.rows or self.cols != another.cols:
            raise ValueError(f'error')
        res = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(another.cols):
                for k in range(another.cols):
                    res[i][j] += another.matrix[i][k] * another.matrix[k][j]
        return res

    @staticmethod
    def subtract_matrix(m1, m2):
        """
         subtract two matrices
        :param m1: first matrix
        :param m2: second matrix
        :return: subtracted matrix
        """
        if m1.rows != m2.rows or m1.cols != m2.cols:
            raise ValueError
        result_matrix = [[m1.matrix[i][j] - m2.matrix[i][j] for j in range(m1.cols)] for i in range(m1.rows)]
        return Matrix(result_matrix)



class Vector(Matrix):
    def __init__(self,vector=None):
        """
         initialize a vector
        :param vector: A list representing the vector
        :return: None
        """
        if vector:
            self.vector = vector
        else:
            self.vector = []

    def __getitem__(self, index):
        return self.vector[index]

    def get_vector_from_input(self):
        """
         get a vector from input
        :return: vector from input
        """
        vector = []
        self.cols = 1
        for i in range(self.rows):
            continue


    def cross_product(self, vector2): #vector product
        """
        cross product two vectors
        :param vector2: second vector
        :return: cross product of two vectors
        """
        x = self[1] * vector2[2] - self[2] * vector2[1]
        y = -1*(self[0] * vector2[2] - self[2] * vector2[0])
        z = self[0] * vector2[1] - self[1] * vector2[0]
        res = [x,y,z]
        return res
    def dot_product(self,vector2):   #scalar product
        """
         dot product of two vectors
         :param vector1: first vector
        :param vector2: second vector
        :return: dot product of two vectors
        """
        return self[0]*vector2[0]+self[1]*vector2[1]+self[2]*vector2[2]



class MyAI(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(MyAI, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.flatten(x)
        return self.dense(x)

import tensorflow as tf
import numpy as np

prompts = ["dot product of two vectors", "add two matrices", "subtract two matrices", "multiply two matrices",
           "find determinant of  matrix","find invert matrix"]
labels = [1, 1, 1, 1,1,1]
vocab = {word: i + 1 for i, word in enumerate(set(" ".join(prompts).split()))}
vocab_size = len(vocab) + 1  # +1 для паддинга
sequences = [[vocab[word] for word in prompt.split()] for prompt in prompts]
max_length = max(len(seq) for seq in sequences)
padded_sequences = [seq + [0] * (max_length - len(seq)) for seq in sequences]
embedding_dim = 8
model = MyAI(vocab_size, embedding_dim)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
X = np.array(padded_sequences)
y = np.array(labels)

model.fit(X, y, epochs=10)




import random

def generate_random_matrix(order, min_val=1, max_val=5):
    return [[random.randint(min_val, max_val) for _ in range(order)] for _ in range(order)]
def execute_command(command):
    if "Hello" in command:
        order = 10
        random_matrix = generate_random_matrix(order)
        matrix = Matrix(random_matrix)


        det = matrix.determinant()
        print("Random matrix:")
        for row in random_matrix:
            print(row)
        print("Determinant of matrix:", det)


        try:
            inverse_matrix = matrix.invert_matrix()
            print("Invert Matrix:")
            for row in inverse_matrix.matrix:
                print(row)


            multiplied_matrix = matrix * matrix
            print("Matrix in 2 degree:")
            for row in multiplied_matrix.matrix:
                print(row)

        except ValueError as e:
            print(e)

    elif "determinant" in command:
        matrix = Matrix()
        result = matrix.determinant()
        print("determinant:", result)
    elif "invert" in command:
        matrix = Matrix()
        try:
            result = matrix.invert_matrix()
            print("invert:")
            for row in result.matrix:
                print(row)
        except ValueError as e:
            print(e)
    elif "dot product" in command:
        vec1 = list(map(int, input("gimme vector(ex: 1 2 2(with space)): ").split()))
        vec2 = list(map(int, input("gimme vector(ex: 1 2 2(with space)): ").split()))
        vector1 = Vector(vec1)
        vector2 = Vector(vec2)
        result = vector1.dot_product(vector2)
        print("dot product:", result)
    elif "add" in command:
        matrix1 = Matrix()
        matrix2 = Matrix()
        result = Matrix.add_matrix(matrix1, matrix2)
        print("Result:")
        for row in result:
            print(row)
    elif "subtract" in command:
        matrix1 = Matrix()
        matrix2 = Matrix()
        result = Matrix.subtract_matrix(matrix1, matrix2)
        print("Result:")
        for row in result:
            print(row)
    elif "multiply" in command:
        matrix1 = Matrix()
        matrix2 = Matrix()
        result = matrix1 * matrix2
        print("Result:")
        for row in result:
            print(row)

    else:
        print('Unknown command')

while True:
    user_input = input("Enter your command (or type 'Hello' to see examples of commands): ")
    execute_command(user_input)
