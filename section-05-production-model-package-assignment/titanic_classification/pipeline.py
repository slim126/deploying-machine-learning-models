from feature_engine.encoding import RareLabelEncoder, OneHotEncoder
from feature_engine.imputation import AddMissingIndicator, CategoricalImputer, MeanMedianImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from titanic_classification.config.core import config
from titanic_classification.processing import features as pp


titanic_pipe = Pipeline(
    [

        # ===== IMPUTATION =====
        # impute categorical variables with string 'missing'
        ('categorical_imputation',
         CategoricalImputer(imputation_method='missing', fill_value="missing", return_object=True,
                            variables=config.model_config.categorical_vars + config.model_config.cabin_vars)),

        # add missing indicator to numerical variables
        ('missing_indicator', AddMissingIndicator(missing_only=True, variables=config.model_config.numerical_vars)),

        # # impute numerical variables with the median
        ('median_imputation', MeanMedianImputer(imputation_method='median', variables=config.model_config.numerical_vars)),

        # Extract first letter from cabin
        ('extract_letter', pp.ExtractLetterTransformer(variables=config.model_config.cabin_vars)),

        # == CATEGORICAL ENCODING ======
        # remove categories present in less than 5% of the observations (0.05)
        # group them in one category called 'Rare'
        ('rare_label_encoder',
         RareLabelEncoder(tol=0.05, n_categories=1, replace_with="Rare", variables=config.model_config.categorical_vars + config.model_config.cabin_vars)),

        # encode categorical variables using one hot encoding into k-1 variables
        ('categorical_encoder', OneHotEncoder(variables=config.model_config.categorical_vars + config.model_config.cabin_vars)),

        # # scale using standardization
        ('scaler', StandardScaler()),

        # logistic regression (use C=0.0005 and random_state=0)
        ('Logit', LogisticRegression(C=0.0005, random_state=0)),
])
