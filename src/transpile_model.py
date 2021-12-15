import joblib

def get_coef_from_model(path_model):
    model = joblib.load(path_model)
    thetas = model.coef_
    return thetas

def get_thetas(thetas):
    thetas_list = "{"
    n_thetas = len(thetas)
    for i in range(n_thetas - 1):
        thetas_list += str(thetas[i])
        thetas_list += ","
    thetas_list += str(thetas[n_thetas - 1])
    thetas_list += "}"
    return thetas_list



def linear_regression_prediction():
    coeffs = get_coef_from_model("../model/model_linear_regression.joblib")

    thetas = coeffs
    thetas_list = get_thetas(thetas)
    n_thetas = len(thetas)
    predict_function = f"""
    #include <stdio.h>
    #include <stdlib.h>
    float linear_regression_prediction(float* features, int n_features)
    {{
        int n_thetas = {n_thetas};
        float thetas[] = {thetas_list};
        float sumX = 0;
        float sumX2 = 0;
        float sumY = 0; 
        float sumXY = 0;
        for (int i = 0; i < n_thetas; i++)
        {{
            sumX = sumX + thetas[i];
            sumX2 = sumX2 + thetas[i] * thetas[i];
            sumY = sumY + features[i];
            sumXY = sumXY + thetas[i] * features[i];
        }}
        float res_tmp = (n_thetas * sumXY - sumX * sumY);
        float res = (sumY - res_tmp * sumX) / n_thetas;
        return res;
    }}
    """
    return predict_function


def logistic_regression_prediction():
    coeffs = get_coef_from_model("../model/model_logistic_regression.joblib")
    thetas = coeffs[0]
    thetas_list = get_thetas(thetas)
    predict_function = f"""
    #include <stdio.h>
    #include <stdlib.h>

    float factorial(float x)
    {{
        if (x <= 1) return 1;
        else return x * factorial(x - 1);
    }}

    int pow_tmp(float x, int y)
    {{
        if (x == 0)
            return 0;
        int tot = 1;
        for (int i = 0; i < y; i++)
            tot *= x;
        return tot;
    }}

    float exp_approx(float x, int n_term)
    {{
        float tot = 0;
        for (int i = 0; i <= n_term; i++)
        {{
            tot += (pow_tmp(x, i) / factorial(i));
        }}
    
        return tot;
    }}

    float sigmoid(float x)
    {{
        return 1/ (1 + exp_approx(-x, 10));
    }}

    float linear_regression_prediction(float* features, float* thetas, int n_thetas)
    {{
        float pred = thetas[0];
        
        for (int i = 1; i < n_thetas; i++)
        {{
        pred += features[i - 1] * thetas[i];
        }}
        return pred;
    }}

    float logistic_regression_prediction(float* features, int n_parameter)
    {{
        float thetas[] = {thetas_list};
        float pred = sigmoid(linear_regression_prediction(features, thetas, n_parameter));
        if (pred <= 0.5)
            return 0.0;
        else
            return 1.0;
    }}
    """
    return predict_function

def create_main_function_linear():
    main_function = """
    int main(int argc, char *argv[])
    {{
        int n_features = argc - 1;
        char *feature_1 = argv[1];
        char *feature_2 = argv[2];
        char *feature_3 = argv[3];
        char *feature_4 = argv[4];
        float features[] = { atof(feature_1), atof(feature_2), atof(feature_3),
        atof(feature_4)};
        printf(\"%f\\n", linear_regression_prediction(features, n_features));
    }}
    """
    return main_function

def create_main_function_logistic():
    main_function = """
    int main(int argc, char *argv[])
    {{
        int n_features = argc - 1;
        char *feature_1 = argv[1];
        char *feature_2 = argv[2];
        char *feature_3 = argv[3];
        char *feature_4 = argv[4];
        float features[] = { atof(feature_1), atof(feature_2), atof(feature_3),
        atof(feature_4)};
        printf(\"%f\\n", logistic_regression_prediction(features, n_features));
    }}
    """
    return main_function

def main():
    model = input("Regression type ? logistic or linear\n")
    while (model not in ['logistic', 'linear']):
        model = input("Regression type ? logistic or linear\n")
    if (model == 'linear'):
        with open("../result/linear_regression.c", "w") as f:
            f.write(linear_regression_prediction())
            f.write(create_main_function_linear())
    else:
        with open('../result/logistic_regression.c', "w") as f:
            f.write(logistic_regression_prediction())
            f.write(create_main_function_logistic())

if __name__ == '__main__':
    main()
