import h2o
from h2o.automl import H2OAutoML

# Initialize H2O
h2o.init()

# Load the iris dataset
data = h2o.import_file("data.csv")

# Set the response column name
y = "median_house_value"
X = [col for col in data.columns if col != y]

# split the data 
train, valid = data.split_frame(ratios=[0.8], seed=42)

aml = H2OAutoML(max_models=10, seed=42)
aml.train(x=X, y=y, training_frame=train, validation_frame=valid)

leaderboard = aml.leaderboard

# Evaluate the performance of the best model
best_model = aml.leader
predictions = best_model.predict(data)

