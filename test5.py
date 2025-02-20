import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Read your training data
features_train = pd.read_csv("train/X_train.csv")
result_train = pd.read_csv("train/Y_train.csv")

data_train = features_train.copy()
data_train['wear_rate'] = result_train['wear_rate']

cat_cols = ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8']
num_cols = ['num7', 'num1', 'num2', 'num8', 'num3', 'num4', 'num9', 'num10', 'num5', 'num6', 'num11', 'num12']

X = data_train[cat_cols + num_cols]
y = data_train['wear_rate']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create transformers for categorical and numerical features
cat_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

num_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Combine the transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', cat_trans, cat_cols),
        ('num', num_trans, num_cols)
    ]
)

# Build the pipeline with preprocessing and the Linear Regression model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model using R² score on the validation set
score = pipeline.score(X_val, y_val)
print("Validation R² Score:", score)

# --- Predict ---

# Import data test
test_features = pd.read_csv('test/X_public_test.csv')
data_test = test_features.copy()

X_test = data_test[cat_cols + num_cols]

# Predict
predictions = pipeline.predict(X_test)

# Show prediction
print("Predictions on test file: ", predictions)

# Save prediction to csv
output = pd.DataFrame(predictions, columns=['wear_rate'])
output.to_csv('predictions.csv', index=False)