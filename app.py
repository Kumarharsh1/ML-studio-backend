from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import traceback
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score, mean_absolute_error, silhouette_score
)
import datetime
import secrets
import string

app = Flask(__name__)
# Updated CORS to allow all origins and handle preflight
CORS(app, origins=["http://127.0.0.1:5500", "http://localhost:5500", "http://127.0.0.1:5501", "http://localhost:5501", "*"], supports_credentials=True)

# Generate a secure secret key
app.secret_key = secrets.token_hex(16)

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration - Use absolute paths
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

print("=" * 60)
print("🚀 ML STUDIO BACKEND STARTING...")
print(f"📁 Base directory: {BASE_DIR}")
print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
print("=" * 60)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def read_file(filepath, filename):
    """Read CSV or Excel file with multiple encoding attempts"""
    try:
        if filename.lower().endswith('.csv'):
            try:
                return pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    return pd.read_csv(filepath, encoding='latin1')
                except UnicodeDecodeError:
                    return pd.read_csv(filepath, encoding='cp1252')
        else:
            return pd.read_excel(filepath)
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        raise e

def clean_data(df):
    """Clean the dataset"""
    original_shape = df.shape
    
    # Remove duplicates
    df = df.drop_duplicates()
    duplicates_removed = original_shape[0] - df.shape[0]
    
    # Handle missing values
    missing_info = {}
    for column in df.columns:
        missing_count = df[column].isnull().sum()
        if missing_count > 0:
            if df[column].dtype in ['int64', 'float64']:
                df[column].fillna(df[column].median(), inplace=True)
            else:
                mode_value = df[column].mode()
                if not mode_value.empty:
                    df[column].fillna(mode_value[0], inplace=True)
                else:
                    df[column].fillna('Unknown', inplace=True)
            missing_info[column] = int(missing_count)
    
    # Remove columns with all null values
    df = df.dropna(axis=1, how='all')
    
    return df, duplicates_removed, missing_info

def prepare_data(df, target_column=None):
    """Prepare data for machine learning"""
    try:
        if target_column is None:
            # Try to find a good target column (last column or column with less unique values)
            for col in reversed(df.columns):
                if df[col].dtype in ['int64', 'float64'] or len(df[col].unique()) < 50:
                    target_column = col
                    break
            if target_column is None:
                target_column = df.columns[-1]
        
        df_encoded = df.copy()
        
        # Encode categorical columns
        for column in df_encoded.select_dtypes(include=['object']).columns:
            if column != target_column:
                le = LabelEncoder()
                df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
        
        # Encode target if categorical
        if df_encoded[target_column].dtype == 'object':
            le = LabelEncoder()
            df_encoded[target_column] = le.fit_transform(df_encoded[target_column].astype(str))
        
        X = df_encoded.drop(columns=[target_column])
        y = df_encoded[target_column]
        
        # Ensure all X columns are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, target_column
        
    except Exception as e:
        print(f"Error in prepare_data: {str(e)}")
        traceback.print_exc()
        raise e

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    """Health check endpoint"""
    if request.method == 'OPTIONS':
        return '', 200
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "uploads_dir_exists": os.path.exists(app.config['UPLOAD_FOLDER'])
    })

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    """Handle file uploads"""
    if request.method == 'OPTIONS':
        return '', 200
        
    print("\n" + "="*50)
    print("📤 UPLOAD REQUEST RECEIVED")
    print("="*50)
    
    if 'file' not in request.files:
        print("❌ No file part in request")
        return jsonify({'success': False, 'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        print("❌ No file selected")
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        print(f"❌ File type not allowed: {file.filename}")
        return jsonify({'success': False, 'error': 'File type not allowed. Please upload CSV, XLSX, or XLS files.'}), 400
    
    try:
        # Generate unique filename
        original_filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        random_string = ''.join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(6))
        filename = f"{timestamp}_{random_string}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file
        file.save(filepath)
        print(f"✅ File saved: {filename}")
        print(f"   Path: {filepath}")
        
        # Read file
        df = read_file(filepath, filename)
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"   Columns: {', '.join(df.columns.tolist())}")
        
        # Clean data
        df_clean, duplicates_removed, missing_info = clean_data(df)
        print(f"✅ Data cleaned: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        if duplicates_removed > 0:
            print(f"   Removed {duplicates_removed} duplicates")
        if missing_info:
            print(f"   Fixed missing values in {len(missing_info)} columns")
        
        # Save cleaned data
        cleaned_filename = f"cleaned_{filename.rsplit('.', 1)[0]}.csv"
        cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cleaned_filename)
        df_clean.to_csv(cleaned_filepath, index=False)
        print(f"✅ Cleaned data saved: {cleaned_filename}")
        
        # Get column data types
        column_types = {}
        numeric_cols = []
        categorical_cols = []
        
        for col in df_clean.columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                column_types[col] = 'numeric'
                numeric_cols.append(col)
            else:
                column_types[col] = 'categorical'
                categorical_cols.append(col)
        
        # Dataset info
        dataset_info = {
            'filename': cleaned_filename,
            'original_filename': original_filename,
            'upload_filename': filename,
            'rows': int(len(df_clean)),
            'columns': int(len(df_clean.columns)),
            'columns_list': df_clean.columns.tolist(),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'column_types': column_types,
            'duplicates_removed': int(duplicates_removed),
            'missing_values': missing_info,
            'total_missing': sum(missing_info.values()) if missing_info else 0,
            'memory_usage': f"{df_clean.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            'timestamp': timestamp
        }
        
        print(f"\n✅ UPLOAD COMPLETE:")
        print(f"   - Rows: {dataset_info['rows']:,}")
        print(f"   - Columns: {dataset_info['columns']}")
        print(f"   - Numeric: {len(numeric_cols)}")
        print(f"   - Categorical: {len(categorical_cols)}")
        print("="*50 + "\n")
        
        return jsonify({
            'success': True,
            'filename': cleaned_filename,
            'original_filename': original_filename,
            'info': dataset_info,
            'message': 'File uploaded and processed successfully'
        })
        
    except Exception as e:
        print(f"\n❌ UPLOAD ERROR: {str(e)}")
        traceback.print_exc()
        print("="*50 + "\n")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_data():
    """Run machine learning analysis on uploaded dataset"""
    if request.method == 'OPTIONS':
        return '', 200
        
    print("\n" + "="*50)
    print("🧠 ANALYSIS REQUEST RECEIVED")
    print("="*50)
    
    try:
        data = request.get_json()
        filename = data.get('filename')
        algorithms = data.get('algorithms', [])
        
        if not filename:
            print("❌ No dataset specified")
            return jsonify({'success': False, 'error': 'No dataset specified'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            print(f"❌ Dataset not found: {filename}")
            return jsonify({'success': False, 'error': 'Dataset not found'}), 404
        
        # Read file
        print(f"📊 Loading dataset: {filename}")
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        results = []
        
        for algorithm in algorithms:
            if algorithm in ['clean_data', 'remove_outliers']:
                continue
                
            print(f"\n▶️ Running: {algorithm}")
            
            try:
                X_train, X_test, y_train, y_test, scaler, target = prepare_data(df.copy())
                
                # Determine if classification or regression
                unique_targets = len(np.unique(y_train))
                is_classification = unique_targets < 20 and algorithm not in ['linear_regression']
                
                # Select model
                if algorithm == 'random_forest':
                    if is_classification:
                        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                        print(f"   Using RandomForestClassifier")
                    else:
                        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                        print(f"   Using RandomForestRegressor")
                        
                elif algorithm == 'decision_tree':
                    model = DecisionTreeClassifier(random_state=42)
                    print(f"   Using DecisionTreeClassifier")
                    
                elif algorithm == 'knn':
                    n_neighbors = min(5, len(X_train))
                    model = KNeighborsClassifier(n_neighbors=n_neighbors)
                    print(f"   Using KNeighborsClassifier (k={n_neighbors})")
                    
                elif algorithm == 'svm':
                    model = SVC(kernel='rbf', random_state=42, probability=True)
                    print(f"   Using SVC")
                    
                elif algorithm == 'linear_regression':
                    model = LinearRegression()
                    print(f"   Using LinearRegression")
                    
                elif algorithm == 'logistic_regression':
                    model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
                    print(f"   Using LogisticRegression")
                    
                elif algorithm == 'naive_bayes':
                    model = GaussianNB()
                    print(f"   Using GaussianNB")
                    
                elif algorithm == 'clustering':
                    n_clusters = min(5, len(X_train))
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = model.fit_predict(X_train)
                    silhouette = silhouette_score(X_train, clusters) if len(np.unique(clusters)) > 1 else 0
                    
                    results.append({
                        'algorithm': 'K-Means Clustering',
                        'silhouette_score': round(float(silhouette), 4),
                        'clusters_created': int(len(np.unique(clusters))),
                        'n_clusters': n_clusters,
                        'model_type': 'clustering'
                    })
                    print(f"   ✅ Completed - Silhouette: {silhouette:.4f}")
                    continue
                    
                else:
                    print(f"   ⚠️ Algorithm not recognized: {algorithm}")
                    continue
                
                # Train model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                algorithm_name = algorithm.replace('_', ' ').title()
                
                # Calculate metrics
                if algorithm in ['linear_regression'] or (algorithm not in ['logistic_regression'] and not is_classification):
                    # Regression metrics
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    
                    results.append({
                        'algorithm': algorithm_name,
                        'r2_score': round(float(r2), 4),
                        'mse': round(float(mse), 4),
                        'mae': round(float(mae), 4),
                        'rmse': round(float(rmse), 4),
                        'model_type': 'regression',
                        'target_column': target
                    })
                    print(f"   ✅ Completed - R²: {r2:.4f}, RMSE: {rmse:.4f}")
                    
                else:
                    # Classification metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    results.append({
                        'algorithm': algorithm_name,
                        'accuracy': round(float(accuracy), 4),
                        'f1_score': round(float(f1), 4),
                        'precision': round(float(precision), 4),
                        'recall': round(float(recall), 4),
                        'model_type': 'classification',
                        'target_column': target,
                        'classes': int(unique_targets)
                    })
                    print(f"   ✅ Completed - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)[:100]}")
                results.append({
                    'algorithm': algorithm.replace('_', ' ').title(),
                    'error': str(e)[:200],
                    'model_type': 'failed'
                })
        
        print(f"\n✅ ANALYSIS COMPLETE: {len([r for r in results if 'error' not in r])} successful, {len([r for r in results if 'error' in r])} failed")
        print("="*50 + "\n")
        
        return jsonify({
            'success': True,
            'results': results,
            'total_algorithms': len(results),
            'successful': len([r for r in results if 'error' not in r])
        })
        
    except Exception as e:
        print(f"\n❌ ANALYSIS ERROR: {str(e)}")
        traceback.print_exc()
        print("="*50 + "\n")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_columns/<filename>', methods=['GET', 'OPTIONS'])
def get_columns(filename):
    """Get column information for a dataset"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        # Read first 1000 rows for analysis
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath, nrows=1000)
        else:
            df = pd.read_excel(filepath, nrows=1000)
        
        columns = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            col_type = 'numeric' if dtype in ['int64', 'float64'] else 'categorical'
            unique_values = len(df[col].dropna().unique()) if len(df) > 0 else 0
            
            columns.append({
                'name': col,
                'dtype': dtype,
                'type': col_type,
                'unique_count': int(unique_values),
                'null_count': int(df[col].isnull().sum())
            })
        
        return jsonify({
            'success': True, 
            'columns': columns,
            'total_columns': len(columns),
            'filename': filename
        })
        
    except Exception as e:
        print(f"Error getting columns: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/dataset_info/<filename>', methods=['GET', 'OPTIONS'])
def dataset_info(filename):
    """Get comprehensive dataset information"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        stats = {
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'missing_values': int(df.isnull().sum().sum()),
            'missing_percentage': round(float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100), 2),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            'duplicates': int(df.duplicated().sum()),
            'size': f"{os.path.getsize(filepath) / 1024 / 1024:.2f} MB"
        }
        
        return jsonify({
            'success': True,
            'filename': filename,
            'shape': [int(df.shape[0]), int(df.shape[1])],
            'columns': df.columns.tolist(),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'stats': stats,
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/delete/<filename>', methods=['DELETE', 'OPTIONS'])
def delete_dataset(filename):
    """Delete a dataset file"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"✅ Deleted: {filename}")
            
            # Also try to delete cleaned version
            cleaned_path = filepath.replace('.csv', '_cleaned.csv')
            if os.path.exists(cleaned_path):
                os.remove(cleaned_path)
                
            return jsonify({'success': True, 'message': 'File deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'File not found'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/list_datasets', methods=['GET', 'OPTIONS'])
def list_datasets():
    """List all uploaded datasets"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        files = []
        for f in os.listdir(app.config['UPLOAD_FOLDER']):
            if f.endswith('.csv') or f.endswith('.xlsx') or f.endswith('.xls'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], f)
                files.append({
                    'name': f,
                    'size': f"{os.path.getsize(filepath) / 1024 / 1024:.2f} MB",
                    'modified': datetime.datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                })
        
        return jsonify({
            'success': True,
            'files': files,
            'count': len(files)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 STARTING ML STUDIO BACKEND SERVER")
    print("="*60)
    print(f"📍 URL: http://localhost:5000")
    print(f"📍 Upload endpoint: http://localhost:5000/upload")
    print(f"📍 Health check: http://localhost:5000/health")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')