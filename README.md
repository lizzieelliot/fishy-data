# fishy-data

The purpose of this report is to analyse the economic and fish datasets and to develop an effective model to predict the net profits of a fishing vessel for that year. Both of the datasets are small with the economic dataset having 1432 observations and the fish data set having 2497 observations.

The economic dataset, which contains data on each fishing boat, was collected by fisheries economic survey. The fish dataset contains all the species of fish and their corresponding revenues.

## Cleaning the dataset
The economic dataset was examined first. Initially there were 39 variables, 19 of which were accounting variables. These accounting variables have no influence on the response variable and have no predictive power whatsoever. They are merely a linear combination of net profits. If a new observation were to join the dataset and it contained these 19 accounting variables, a model would not be necessary as net profits would be simply computed through the linear combination of the accounting variables. It was, therefore, necessary to drop the following variables.

The remaining 19 variables were then checked for zero variance using the nearZeroVar() function in the caret package. Zero variance variables are not only non-informative but can prevent fitting an effective model. In this case there were no variables that had zero variance.

The data was then checked for missing entries using the aggr() function in the VIM package. This dataset has a large proportion of missing entries. If a variable has a high proportion of missing data, there may be flaws in how this data was collected. It appears that a questionnaire was used to collect economic dataset. Possibly questions may have been unclear or might not have applied to all participants.

Those variables with missing entries required further analysis. The percentage of data missing from each variable was calculated and a cut off of 30% missing data was applied. If this approach had not been taken variables such as Total.Jobs would have been used to build a model that would not have been representative of 30% of the sample. In addition, Fishing.Income was removed as later the fish dataset would be merged with the economic dataset, rendering this variable useless. This resulted in the following variables being dropped.

It is important to note that not all missing data entries are detected through the aggr() function, as they may be incorrectly entered as zero. This is in the case of Capacity..GT., a measure of capacity in gross tonnage; as volume of a vessel cannot be zero it is assumed that zeros are missing data. Zeros were changed to NA’s in this case.

With three remaining variables, Capacity.Index, Capacity..GT, Size.Index, that had missing entries of roughly 3%, it was thought best to compute surrogates. This was done using the package mice(), which aims to create multiple imputations for multivariate missing data. CART method was used. The results can be seen from figure 2.

The remaining 15 variables were checked for high correlation to (i) get an overview of the data and (ii) examine whether or not more variables could be eliminated. There were a number of variables that are highly correlated, indicating multicollinearity. Having more than one variable that measures the same thing may result in models that are overly biased towards these variables. Keeping such variables in the dataset would not only over complicate the model with useless variables, but would also reduce the accuracy of the models.

There appeared to be many variables that measure size namely: Size.Category, Length, Size.Index, Capacity..GT, Capacity.Index. Only one of these variables is needed in practice. A continuous variable was chosen, Capacity..GT, as it was believed that it would be more accurate. Using this approach, a total of 7 variables were removed.

The economic and fish datasets were merged by ‘vid’ and ‘Year’, allowing the removal of these variables afterwards as they have no predictive properties. Reference.Number was also removed. This resulted in a dataset with 108 variables as fish were explicitly requested to be used in the model by the client. Both datasets only had 756 corresponding observations resulting in a very small, but complex, dataset to build a predictive regression model on.
 
## Modelling the data

Before building any models, the data was split into a training and testing. A number of ratios were tried and tested which resulted in a 70:30 split being chosen. This was also deemed appropriate as the sample size is very small having 756 observations. If the training set was too small the models would be less accurate at predicting. If the training set was too big the models would risk overfitting. Overfitting occurs when the model captures the noise of the data, which results in the model fitting the training dataset too well.

For this dataset the following models were used;
(v) Single decision tree
(vi) Random Forest
(vii) Boosting model

### Single decision tree
To construct a single decision tree model the function rpart() was used on the training set. rpart() aims to classify observations by recursively splitting the population into sub- populations, which are then split again and so on. The initial tree was created using the default complexity parameter of 0.01. A smaller cp was not chosen at the beginning as to avoid building a model that was too complex. The optimum complexity parameter was then calculated by plotting the cp’s against the relative error using plotcp() function. Complexity parameter controls the size of the tree. If the tree is too large the model may over-fit. A cp of 0.01699023 was chosen, which can be seen in table 3.

To measure the accuracy of the models the correlation between the predicted net profit and the actual net profit was calculated in. In addition, three measures of error were calculated: mean squared error, mean absolute error and root mean squared error. RMSE has the advantage of penalising large errors more, so can be an appropriate measure of error if we are concerned with predictions that are very different to the actual net profits of the vessels.
It can be seen that while the correlation is 2% higher, by using the optimum cp for the single decision tree, the predictions have become less accurate; this may be because a dataset with many variables (108) needs a complex tree to make accurate predictions. A much smaller cp was used to build a decision tree in an effort to improve the accuracy of the predictions. However, as can be seen from table 5, little improvement was made. The marginal improvement in the correlation, MSE, MAE and RMSE, does not justify building a tree that is far more complex; therefore, it was thought sensible to stay with the default cp of 0.01 for ease of computation, despite being a weak predictor.

The optimum single decision tree, of size 3; while it is a weak predictor, it is easy to understand. It can be seen that Horse Mackerel is a strong predictor, along with Capacity..GT. and Albacore. If you fish for a lot of Horse Mackerel and your capacity is larger than 1449 GT you are more likely to have large net profits.

### Random Forest
Random forest is another ensemble method which is based on trees. It aims to randomly select samples of m from M variables so that certain variables, which are sometimes not picked due to being overpowered by dominant variables, are still explored.

Firstly, the tuneRF() function was used to find the optimum mtry parameter. Mtry is the number of variables sampled at each split. The optimum mtry can be identified by plotting the mtry against the Out of Bag Error, which is the predicted error estimate of the random forest. The optimum mtry that was chosen is 35.

The initial random forest was grown using the randomForest() function consisting of 500 trees as a starting point. Then error rate of the model was then plotted against the number of trees. It can clearly be seen that the error drops significantly until a minimum of approximately 70 and rises slowly before levelling off inbetween. For this reason is sensible to grow a random forest consisting of approximately 70 trees.

Both random forest models with 500 and 70 trees respectively produce similar plots between their predicted values for net profits and the actual values for net profits. The random forest model with 70 trees minutely outperforms the random forest model of 500 trees. Therefore, it is wise to build a model with 70 tree to not only improve the effectiveness, but mainly to reduce computational time and complexity of the model. Despite this, neither random forests prove to be good predictors of net profit.

An added benefit of using random forest as a predictive model is that it gives additional information on the variables including variable importance. The mean decrease in accuracy plot (%InceMSE) measures the decrease in accuracy of the random forest if that variable is excluded from the model. Therefore, the higher the mean decrease in accuracy, the higher the importance of the variable. Mean decrease Gini (IncNodePurity) is a measure of how each variable contributes to the sameness of each node and leaves in the random forest. Variables that produce nodes with high impurity are given a higher mean decrease Gini and are, therefore, more important.

The importance() function shows that segment is the most influential variable in predicting a fishing vessel’s net profits for the year. It follows that these fishing methods, such as pelagic trawlers, are the greatest determining feature of a fishing vessel’s net profit for the year.

It is interesting to compare the two graphs of mean decrease in accuracy and mean decrease Gini. One peculiar feature is the differing features of the variable John.dory on each graph. This may suggest that the random forest is not a stable model and by simply changing the seed, predictive variables, particularly those which are fish, variable importance may change. This is a suggestion that the fish variables have little predictive power.

Variables can be examined further using partial dependency plot, which can be produced using the partialPlot() function. Partial dependency plot shows how much influence a predictive variable has on the response variable, holding all other variables constant.

Capacity..GT., to be a variable with a high importance value. It seems to have a low influence on the response variable at 0 GT, however increases steadily until 600 GT and continues to have a high positive influence on net profits thereafter.

The second variable plotted, Total Active Vessels in Segment has a the third highest importance value. It has a medium influence on the response variable at 0; this may be that as there are no active vessels it is common then to have a predictable net profit unless there are unforeseen circumstances, such as legal bills. The influence falls dramatically before climbing just as suddenly around 50 active vessels. Thereafter it has a very high influence on the fishing vessel’s net profit for that year.

Overall the optimised random forest gives weak predictions of net profit for a fishing vessel.

### Boosting

Boosting is another ensemble method that aims to reduce bias. In this problem, gradient boosting was used. In gradient boosting many models are trained one after the other, improving each time by minimising the loss function. To build the boosting model the gbm() function was used. The Gaussian method was used as this is a continuous regression problem.

Initially a boosting model was built with 500 trees, shrinkage (the rate of learning) of 0.01 and number of terminal nodes in the tree at 5, to avoid building an overly complex model.

Firstly, the number of trees was adjusted to optimise the model. Very little difference was made in terms of the correlation, MSE, MAE and RMSE. Despite all 4 criteria do get better as the trees grow, the marginal benefit of growing trees between 500 and 2000 is negligible. Therefore, to avoid building an overcomplicated model, 500 trees was deemed adequate.

The shrinkage was altered, however values larger and smaller than 0.01 produced less accurate predictions.
Next the number of terminal nodes in the tree was altered; 3 appears to be the optimum number of terminal nodes of the boosting tree, producing a correlation between the predicted net profit and the actual net profit of 71%.

The boosting model is the most effective in predicting the net profits for that year of a fishing vessel. It outperforms the single decision tree and random forest marginally. It has a correlation of 71% but also has very high MSE, MAE and RMSE, indicating that in fact this is not an effective regression model. This may be partially attributed to the inclusion of the fish dataset that is believed to have little predictive power. It comes as a surprise that the single decision tree outperformed the random forest. It also has the added benefit of being interpreted easily unlike the ‘black box’ ensembles. In conclusion, it is advisable that none of these models should be relied on for an accurate prediction of net profit.

If further analysis was to be completed on this dataset, the fish dataset would not be included, as not only does it have little predictive power, it may have affected the performance of the models. It is believed that the prediction of net profit for fishing vessels is a forecasting problem and methods such as ARIMA should be employed for further analysis.

