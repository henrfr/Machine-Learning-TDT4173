# tdt-4173-revenue

**Ideas for bustop dataset:**
- Impute importance from description. 
- Use https://vegkart.atlas.vegvesen.no/#kartlag:geodata/@270755,7039648,14/hva:!(id~487)~ to gain insight

**Ideas for good features**
- Population density
- Distance to/importance of closest bus stop

**28.09**
- busstop id is very correlated to revenue
- try to predict categories to see if there is a difference
- try to find city centrums
- group demographics by district/muncipality
- try xgboost


**06.10**
- Set up a stacking model with all models
- Set up a stacking model with the best models
- Find best features via `grid_search.best_estimator_.feature_importances_`.
- The model for delivering should
  - Be run 4/5 times with a K-fold split and then for every predicted point take an average
