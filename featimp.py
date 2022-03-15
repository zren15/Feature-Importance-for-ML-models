def spearman_rank(df,y,X):
    corr = {}
    cols = [cols for cols in X.columns]
    y = df["PRICE"] 
    n = y.count()
    y.index = np.arange(n)
    y_rank = np.array(y.rank())
    y_std = np.array(y.rank()).std()
    for i in cols:
        x= df[i]
        x.index = np.arange(n)
        x_rank = np.array(x.rank())
        cov = np.cov(x_rank,y_rank)
        x_std = np.array(x.rank()).std()
        p = cov/(x_std * y_std)
        corr[i] = abs(p[0][1])
    return dict(sorted(corr.items(), key = lambda x: x[1], reverse = True)) 

def pca_importance(x_train):
    pca = PCA()
    
    sc = StandardScaler()
    x_train_std = sc.fit_transform(x_train)
    
    pca = pca.fit(x_train_std)
    
    # Change pcs components ndarray to a dataframe
    importance_df  = pd.DataFrame(pca.components_)
    
    # Assign columns
    importance_df.columns  = x_train.columns

    # Change to absolute values
    importance_df =importance_df.apply(np.abs)

    # Transpose
    importance_df=importance_df.transpose()
    # Change column names again

    ## First get number of pcs
    num_pcs = importance_df.shape[1]

    ## Generate the new column names
    new_columns = [f'PC{i}' for i in range(1, num_pcs + 1)]

    ## Rename
    importance_df.columns = new_columns
    
    ## Sort the first principle component 
    pc1_top_features = importance_df['PC1'].sort_values(ascending = False)
    pc1_top_features = pc1_top_features.to_dict()
    # Return importance df
    return pc1_top_features

def calculate_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter
):
    # train model
    model.fit(X, y)
    # make predictions for train data and score
    y_hat_initial = model.predict(X)
    score_r2 = r2_score(y, y_hat_initial)
    # calculate permutation importance
    importances = dict(zip(X.columns, value))

    for n in range(n_iter):
        for col in list(X.columns):
            # copy data to avoid using previously shuffled versions
            X_temp = X.copy()

            # shuffle feature_i values
            X_temp[col] = X[col].sample(X.shape[0], replace=True, random_state=random.randrange(0, 2**4)).values

            # make prediction for shuffled dataset
            y_hat = model.predict(X_temp)

            # calculate score
            score_permuted_r2 = r2_score(y, y_hat)

            # calculate delta score
            # better model <-> higher score
            # lower the delta -> more important the feature
            delta_score = score_permuted_r2 - score_r2
            # get absolute value 
            delta_score = abs(delta_score)
            # save result
            importances[col] += delta_score / n_iter

    importances_values = np.array(list(importances.values()))
    return dict(sorted(importances.items(), key = lambda kv:(kv[1], kv[0]),reverse=True))

def dropcol_importances(model, X_train, y_train, X_valid=None, y_valid=None, sample_weights=None):
    if X_valid is None: 
        X_valid = X_train
    if y_valid is None: 
        y_valid = y_train
    model_ = clone(model)
    model_.random_state = 999
    model_.fit(X_train, y_train)
    baseline = model_.score(X_valid, y_valid, sample_weights)
    imp = []
    for col in X_train.columns:
        model_ = clone(model)
        model_.random_state = 999
        model_.fit(X_train.drop(col,axis=1), y_train)
        s = model_.score(X_valid.drop(col,axis=1), y_valid, sample_weights)
        drop_in_score = baseline - s
        imp.append(drop_in_score)
    imp = np.array(imp)
    I = pd.DataFrame(data={'Feature':X_train.columns, 'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    I = I.to_dict()
    return I['Importance']

def shap_importance(model,x_train,y_train):
    model.fit(x_train, y_train)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_train)
    shap_imp_pd=pd.DataFrame(index=data.feature_names, data = np.mean(np.absolute(shap_values), axis = 0), columns=["Importance"])
    shap_imp_pd = shap_imp_pd.sort_values('Importance', ascending=False)
    shap_imp_pd = shap_imp_pd.to_dict()
    sort_importances = shap_imp_pd['Importance']
    return sort_importances

def compare_feature(model, train_df, test_df, feat_imp:dict, metric=mean_squared_error):

    MSE = []
    train = train_df.copy()
    val = test_df.copy()
    for i in range(1, 14):
        model_ = clone(model)
        model_.random_state = 3
        features = [col for col in feat_imp.keys()][:i]
        model_.fit(train.loc[:, features], train['PRICE'])
        predictions = model_.predict(val.loc[:, features])
        mse_valid = metric(val['PRICE'], predictions)
        MSE.append(mse_valid)
    return MSE

def permutation_importances_mse(model, x_train, y_train,  x_test, y_test):
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    baseline = mean_squared_error(y_test, prediction)
    imp = []
    for col in x_test.columns:
        save = x_test[col].copy()
        x_test[col] = np.random.permutation(x_test[col])
        m = mean_squared_error(y_test, model.predict(x_test))
        x_test[col] = save
        imp.append(baseline - m)

def auto_selection(feature_list, x_train, y_train, x_test, y_test):
    model = RandomForestRegressor(n_estimators=100,n_jobs = -1)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    mse = mean_squared_error(y_test, prediction)
    k = len(feature_list)
    new = {}
    feature = []

    least_important_sort = feature_list[::-1] # reverse
    final = feature_list[::-1]
    for i, feat in enumerate(least_important_sort):
        print('drop feature:',feat)
        feature.append(feat)
        x_train= x_train.drop(feat, axis=1)
        x_test = x_test.drop(feat, axis=1)

        model = RandomForestRegressor(n_estimators=100,n_jobs = -1)
        model.fit(x_train, y_train)

        perm_list = permutation_importances_mse(model, x_train, y_train,  x_test, y_test)
        feat = perm_list[-1] # last important feature
        prediction = model.predict(x_test)
        mse = mean_squared_error(y_test, prediction)
        new[i] = mse
        print('MSE:',new[i])
        mse_list = list(new.values())
        min_loc = mse_list.index(min(mse_list))
        if i > 0: 
            if new[i] > new[i-1]:
                print('Stopping iterations as MSE did not decrease')
                print('Drop Feature', feature[:-1])
                break

    return new

def spearman_rank2(y,X):
    corr = {}
    cols = list(X.columns)
    n = y.count()
    y.index = np.arange(n)
    y_rank = np.array(y.rank())
    y_std = np.array(y.rank()).std()
    for i in cols:
        x= X[i] 
        x.index = np.arange(n)
        x_rank = np.array(x.rank())
        x_std = np.array(x.rank()).std()
        p = np.cov(x_rank,y)/np.sqrt(np.var(x_rank)*np.var(y))
        corr[i] = abs(p[0][1])
    return dict(sorted(corr.items(), key = lambda x: x[1], reverse = True)) 

def calculate_variance_Spearman(data):
    info=[]
    for i in range(100):
        sample = data.sample(len(data),replace=True).reset_index(drop=True)
        sample_x= sample.drop(columns = 'PRICE')
        sample_y = sample['PRICE']
        info.append(spearman_rank2(sample_y,sample_x))
    return pd.DataFrame(info) 