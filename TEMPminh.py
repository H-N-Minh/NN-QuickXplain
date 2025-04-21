import pandas as pd


################## count number of rows and collumn in both files

# # arcade_small_conflicts_410.csv
# # Number of rows: 410
# # Number of columns: 48
# # Unique values {np.int64(0), np.int64(1), np.int64(-1)}

# #arcade_small_invalid_confs_410.csv
# # Number of rows: 410
# # Number of columns: 48
# # Unique values{np.int64(1), np.int64(-1)}




# ############################## 
# # show that -1 and 1 in both files must match
# # Read both CSV files
# conflicts_path = "TrainingData/arcade_small_conflicts_410.csv"
# invalid_confs_path = "TrainingData/arcade_small_invalid_confs_410.csv"

# conflicts = pd.read_csv(conflicts_path, header=None)
# invalid_confs = pd.read_csv(invalid_confs_path, header=None)

# # Exclude the first column if needed (uncomment if first column is not data)
# # conflicts = conflicts.iloc[:, 1:]
# # invalid_confs = invalid_confs.iloc[:, 1:]

# # Check for mismatches where conflicts is 1 or -1 but invalid_confs is not the same
# mismatch = ((conflicts == 1) & (invalid_confs != 1)) | ((conflicts == -1) & (invalid_confs != -1))

# # Ignore cells where conflicts is 0
# mismatch = mismatch & (conflicts != 0)

# # Find indices of mismatches
# mismatch_indices = list(zip(*mismatch.to_numpy().nonzero()))

# if mismatch_indices:
#     print("Mismatches found at (row, column):")
#     for idx in mismatch_indices:
#         print(f"Row {idx[0]}, Column {idx[1]}: conflicts={conflicts.iat[idx[0], idx[1]]}, invalid_confs={invalid_confs.iat[idx[0], idx[1]]}")
# else:
#     print("No mismatches found.")


# Compare temp1.csv and TrainingData/arcade_small_conflicts_410.csv for exact match

temp1_path = "temp1.csv"
arcade_conflicts_path = "TrainingData/arcade_small_conflicts_410.csv"

df1 = pd.read_csv(temp1_path, header=None)
df2 = pd.read_csv(arcade_conflicts_path, header=None)

if df1.equals(df2):
    print("The files have exactly the same data.")
else:
    print("The files are different. Differences found at (row, column):")
    diff = df1 != df2
    diff_indices = list(zip(*diff.to_numpy().nonzero()))
    for idx in diff_indices:
        print(f"Row {idx[0]}, Column {idx[1]}: temp1.csv={df1.iat[idx[0], idx[1]]}, arcade_small_conflicts_410.csv={df2.iat[idx[0], idx[1]]}")