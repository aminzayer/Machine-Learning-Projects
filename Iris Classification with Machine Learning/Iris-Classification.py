from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# بارگیری داده‌های Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target

# پیش‌پردازش داده‌ها: مقیاس دهی
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# تقسیم داده‌ها به داده‌های آموزشی و آزمون
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# انتخاب مدل (رگرسیون لجستیک)
model = LogisticRegression(max_iter=1000)

# اعتبارسنجی متقابل (Cross Validation)
cv_scores = cross_val_score(model, X_scaled, y, cv=5)

print("Cross Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# آموزش مدل با داده‌های آموزشی
model.fit(X_train, y_train)

# پیش‌بینی بر روی داده‌های آزمون
y_pred = model.predict(X_test)

# ارزیابی دقت مدل
accuracy = model.score(X_test, y_test)
print("Test Accuracy:", accuracy)
