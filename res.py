import pandas as pd

c_1= pd.read_csv('D:\\wear_main\\logs\\tridet\\2023-07-11_11-40-33\\unprocessed_results\\v_seg_wear_split_1.csv')
c_2= pd.read_csv('D:\\wear_main\\logs\\tridet\\2023-07-11_11-40-33\\unprocessed_results\\v_seg_wear_split_2.csv')
c_3= pd.read_csv('D:\\wear_main\\logs\\tridet\\2023-07-11_11-40-33\\unprocessed_results\\v_seg_wear_split_3.csv')
i_1= pd.read_csv('D:\\wear_main\\logs\\tridet\\2023-07-11_14-50-58\\unprocessed_results\\v_seg_wear_split_1.csv')
i_2= pd.read_csv('D:\\wear_main\\logs\\tridet\\2023-07-11_14-50-58\\unprocessed_results\\v_seg_wear_split_1.csv')
i_3= pd.read_csv('D:\\wear_main\\logs\\tridet\\2023-07-11_14-50-58\\unprocessed_results\\v_seg_wear_split_1.csv')

merged_df1 = pd.concat([c_1, i_1], ignore_index=True)
merged_df2 = pd.concat([c_2, i_2], ignore_index=True)
merged_df3 = pd.concat([c_3, i_3], ignore_index=True)

sorted_df = merged_df1.sort_values(['video-id', 'score'], ascending=[True, False])
sorted_df.to_csv('sorted_file.csv', index=False)
top_2000_df = sorted_df.groupby('video-id').head(2000)
top_2000_df.to_csv('selected_file.csv', index=False)
