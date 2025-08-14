
library(caret)
library(e1071)
library(rpart)
library(rpart.plot)
install.packages("ggplot2")
library(ggplot2)
library(class)  # For KNN

# Load the dataset
diabetes_data <- read.csv(file.choose())
View(diabetes_data)

# Normalize features (excluding the target column)
normalize <- function(x) (x - min(x)) / (max(x) - min(x))
diabetes_data[ ,1:8] <- lapply(diabetes_data[ ,1:8], normalize)

# Convert the target variable to a factor for classification
diabetes_data$Outcome <- as.factor(diabetes_data$Outcome)

# Train-test split
set.seed(123)
trainIndex <- createDataPartition(diabetes_data$Outcome, p = 0.8, list = FALSE)
train_data <- diabetes_data[trainIndex, ]
test_data <- diabetes_data[-trainIndex, ]

# Data frame to store model accuracies
accuracy_results <- data.frame(Model = character(), Accuracy = numeric())

### 1. K-Nearest Neighbors (KNN) ###
k <- 5  # You can adjust this value
knn_predictions <- knn(train = train_data[ ,1:8], test = test_data[ ,1:8], cl = train_data$Outcome, k = k)
cm_knn <- confusionMatrix(knn_predictions, test_data$Outcome)
accuracy_knn <- cm_knn$overall["Accuracy"]
accuracy_results <- rbind(accuracy_results, data.frame(Model = "K-Nearest Neighbors", Accuracy = accuracy_knn))

# Plot for KNN predictions
ggplot(data.frame(Actual = test_data$Outcome, Predicted = knn_predictions), aes(x = Actual, fill = Predicted)) +
  geom_bar(position = "dodge") +
  labs(title = "KNN: Actual vs Predicted", x = "Actual Outcome", y = "Count", fill = "Predicted Outcome") +
  theme_minimal()





#Naive Bayes
nb_model <- naiveBayes(Outcome ~ ., data = train_data)
nb_predictions <- predict(nb_model, test_data)
cm_nb <- confusionMatrix(nb_predictions, test_data$Outcome)
accuracy_nb <- cm_nb$overall["Accuracy"]
accuracy_results <- rbind(accuracy_results, data.frame(Model = "Naive Bayes", Accuracy = accuracy_nb))

# Plot for Naive Bayes predictions
ggplot(data.frame(Actual = test_data$Outcome, Predicted = nb_predictions), aes(x = Actual, fill = Predicted)) +
  geom_bar(position = "dodge") +
  labs(title = "Naive Bayes: Actual vs Predicted", x = "Actual Outcome", y = "Count", fill = "Predicted Outcome") +
  theme_minimal()













### 3. Decision Tree ###
tree_model <- rpart(Outcome ~ ., data = train_data, method = "class")
tree_predictions <- predict(tree_model, test_data, type = "class")
cm_tree <- confusionMatrix(tree_predictions, test_data$Outcome)
accuracy_tree <- cm_tree$overall["Accuracy"]
accuracy_results <- rbind(accuracy_results, data.frame(Model = "Decision Tree", Accuracy = accuracy_tree))

# Plot for Decision Tree predictions
ggplot(data.frame(Actual = test_data$Outcome, Predicted = tree_predictions), aes(x = Actual, fill = Predicted)) +
  geom_bar(position = "dodge") +
  labs(title = "Decision Tree: Actual vs Predicted", x = "Actual Outcome", y = "Count", fill = "Predicted Outcome") +
  theme_minimal()

# Plot the Decision Tree
rpart.plot(tree_model, main = "Decision Tree Structure")





### 4. Linear Regression (Converted for Binary Classification) ###
lm_model <- lm(as.numeric(Outcome) ~ ., data = train_data)
lm_predictions <- predict(lm_model, test_data)
lm_class <- as.factor(ifelse(lm_predictions > 0.5, 1, 0))
cm_lm <- confusionMatrix(lm_class, test_data$Outcome)
accuracy_lm <- cm_lm$overall["Accuracy"]
accuracy_results <- rbind(accuracy_results, data.frame(Model = "Linear Regression", Accuracy = accuracy_lm))

# Plot for Linear Regression predictions
ggplot(data.frame(Actual = as.numeric(test_data$Outcome), Predicted = lm_predictions), aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Linear Regression: Actual vs Predicted", x = "Actual Outcome", y = "Predicted Outcome") +
  theme_minimal()

### Final Comparison of Model Accuracies ###
# Print accuracy statistics for each model
print(accuracy_results)


# Plot model accuracy comparison
ggplot(accuracy_results, aes(x = reorder(Model, Accuracy), y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_text(aes(label = round(Accuracy, 2)), vjust = -0.5, size = 4) +
  labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
  theme_minimal()
