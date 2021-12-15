# Transpiler for C

## How to use
First compute the model.py in the src directory, this will create two models : one for logistic and one for linear regression

Then compute the transpile_model.py, this will create a C file depending of your choice

## C files
Once you have your C file, you can compute it using gcc. It will create a dependance (a.out)

## a.out
Now you have your executable, you can execute it with ./a.out [list_of_features]
