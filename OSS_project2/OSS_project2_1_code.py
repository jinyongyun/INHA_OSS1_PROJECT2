import pandas as pd

# Load the CSV file
df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

print("\n2-1 1st problem : Filter data for the years 2015 to 2018\n")
for year in range(2015, 2019):
    print(f"\nTop 10 in {year} year:")
    year_df = df[df['year'] == year]
    
    hits_top10 = year_df.sort_values(by='H', ascending=False).head(10)[['batter_name', 'H']]
    avg_top10 = year_df.sort_values(by='avg', ascending=False).head(10)[['batter_name', 'avg']]
    hr_top10 = year_df.sort_values(by='HR', ascending=False).head(10)[['batter_name', 'HR']]
    obp_top10 = year_df.sort_values(by='OBP', ascending=False).head(10)[['batter_name', 'OBP']]
    
    print("Top 10 in Hits(안타):")
    print(hits_top10)
    print("\nTop 10 in Batting Average(타율):")
    print(avg_top10)
    print("\nTop 10 in Homerun(홈런):")
    print(hr_top10)
    print("\nTop 10 in On-base Percentage(출루율):")
    print(obp_top10)
    
    
print("\n2-1 2st problem: Filter data for 2018\n") 
df_2018 = df[df['year'] == 2018]

# Sort the DataFrame by 'war' in descending order with each cp
sorted_war = df_2018.sort_values(by=['cp', 'war'], ascending=[True, False])

# Drop duplicate positions, keeping only the first for each cp
highest_war_by_cp = sorted_war.drop_duplicates('cp')[['cp', 'war',  'batter_name']]

print("\nPlayer highest war by cp in 2018:")
print(highest_war_by_cp)
    
    
 
print("\n2-1 3st problem: Among R, H, HR, RBI, SB, war, avg, OBP, and SLG, which has the highest correlation with salary?\n") 
correlation = df[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']].corrwith(df['salary'])
highest_feature = correlation.abs().nlargest(1).index[0]
print(f"\nhighest correlation with salary : {highest_feature}")