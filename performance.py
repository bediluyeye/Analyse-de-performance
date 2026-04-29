import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#Chargement des données et nettoyage
df = pd.read_csv('performance_employes_500_real_names.csv', sep=',')
df = df.dropna()
df = df.drop_duplicates()
print(df.isnull().sum())

###Analyse exploratoire
#Departement performants
best_dep_perf =df.groupby('departement')['performance_score'].mean().sort_values(ascending=False)
print(best_dep_perf)

plt.figure(figsize=(8,5))
plt.bar(best_dep_perf.index, best_dep_perf.values)

plt.title("Performance moyenne par département")
plt.xlabel("Département")
plt.ylabel("Performance moyenne")

plt.xticks(rotation=45)
plt.show()

features = ['heures_travail', 'jours_absence','formations_suivies', 'annees_experience', 'satisfaction_score']

#Verification de toutes les correlations possibles
for col in features:
    sns.regplot(data=df, x=col, y='performance_score')
    plt.title(f"{col} vs performance")
    plt.show()

corr = df.corr(numeric_only=True)["performance_score"][1:6]
# corr_percent = (corr * 100).round(2).astype(str) + " %"
print(corr)

# Teste avec seulement les deux variables les plus prometteuses
target = 'performance_score'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE :", mse)
print("R² :", r2)


resultats = pd.DataFrame({
    "nom": df.loc[y_test.index, "nom"],
    "departement": df.loc[y_test.index, "departement"],
    "Vraie valeur": y_test.values,
    "Prediction": y_pred
})

print(resultats.head())

#visual
import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))

# points
plt.scatter(y_test, y_pred, alpha=0.7)

# ligne parfaite (prédiction idéale)
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))

plt.plot([min_val, max_val], [min_val, max_val], color='red')

plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Réel vs Prédit (Régression Linéaire)")

plt.show()