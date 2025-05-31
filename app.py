import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# Load and label data
def load_data():
    cn = pd.read_csv('Data/cn.csv')
    emci = pd.read_csv('Data/emci.csv')
    lmci = pd.read_csv('Data/lmci.csv')
    mci = pd.read_csv('Data/mci.csv')

    cn['Label'] = 'CN'
    emci['Label'] = 'EMCI'
    lmci['Label'] = 'LMCI'
    mci['Label'] = 'MCI'

    return pd.concat([cn, emci, lmci, mci], ignore_index=True)

# Preprocess features
def preprocess_data(df):
    print("Encoding categorical variables...")

    # Encode Sex
    df['Sex'] = df['Sex'].map({'F': 0, 'M': 1})

    # One-hot encode Description
    df = pd.get_dummies(df, columns=['Description'])

    # Feature matrix X and target y
    X = df.drop(columns=['Subject ID', 'Label'])
    y = LabelEncoder().fit_transform(df['Label'])

    # Standardize Age
    X['Age'] = StandardScaler().fit_transform(X[['Age']])

    # Drop missing values
    X = X.dropna()
    y = y[X.index]  # Align target with feature indices

    return X, y

# Train and evaluate model
def train_model(X, y):
    print("Splitting data and training model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nModel Performance:\n")
    print(classification_report(y_test, y_pred))

    return model

# Main pipeline
if __name__ == '__main__':
    print("Loading data...")
    df = load_data()
    print("Data loaded. Shape:", df.shape)

    X, y = preprocess_data(df)
    print("Data preprocessed. Feature matrix shape:", X.shape)

    model = train_model(X, y)

    # Save model and column names
    joblib.dump(model, 'model.pkl')
    joblib.dump(X.columns.tolist(), 'model_columns.pkl')

    print("✅ Model and column list saved as 'model.pkl' and 'model_columns.pkl'")
