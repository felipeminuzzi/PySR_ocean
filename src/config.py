raw_df_path        = './data/raw/era5_reanalysis_utlimos_dados.nc'
latitude           = -49
longitude          = -31
flag               = False   #true for single point, false for space predict
test_initial_date  = '2021-01-01'
train_initial_date = '2020-01-01'
lat_tst            = -48.5
long_tst           = -30.5
new_train          = False #set true if want to train new model
model_saved        = './outputs/20250602_085104_ht5Fqn'


# array([-48. , -48.5, -49. , -49.5, -50. , -50.5, -51. ])
# array([-32. , -31.5, -31. , -30.5, -30. ])