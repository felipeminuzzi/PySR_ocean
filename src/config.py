raw_df_path        = './data/raw/dados_atlantico2013_2024.nc'
latitude           = -49
longitude          = -31
flag               = False   #true for single point, false for space predict
test_initial_date  = '2021-01-01'
train_initial_date = '2020-01-01'
lat_tst            = -20.0
long_tst           = -25.0
new_train          = False #set true if want to train new model
model_saved        = './outputs/20250912_140050_5n0eWq'
future_predict     = True
region_time        = '2021-04-12 12:00:00'
feature_var        = ['Wave_age','1/t_star','log_tstar','latitude','longitude']
target_var         = ['log_y']

# array([-48. , -48.5, -49. , -49.5, -50. , -50.5, -51. ])
# array([-32. , -31.5, -31. , -30.5, -30. ])
