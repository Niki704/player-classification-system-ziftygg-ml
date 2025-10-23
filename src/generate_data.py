import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load your real data
print("\nLoading real data from Google Form responses...")

try:
    real_data = pd.read_csv('data/raw/google_form_responses.csv')
    print(f"Successfully loaded {len(real_data)} real responses")
except FileNotFoundError:
    print("ERROR: Could not find 'data/raw/google_form_responses.csv'")
    print("Please make sure the file is in the correct location.")
    exit()

# Display basic info
print(f"\nColumns found: {len(real_data.columns)}")
print(f"Rows found: {len(real_data)}")

# Step 2: Data Cleaning and Preparation
print("\nCleaning data...")

# Rename columns for easier handling
real_data_clean = real_data.copy()
real_data_clean.columns = [
    'timestamp', 'ign', 'uid', 'experience_level', 'preferred_mode',
    'daily_play_time', 'codm_experience', 'mp_legendary_streak',
    'br_legendary_streak', 'mp_kd_ratio', 'interest_rating'
]

# Convert numeric columns
real_data_clean['experience_level'] = pd.to_numeric(real_data_clean['experience_level'], errors='coerce')
real_data_clean['mp_legendary_streak'] = pd.to_numeric(real_data_clean['mp_legendary_streak'], errors='coerce')
real_data_clean['mp_kd_ratio'] = pd.to_numeric(real_data_clean['mp_kd_ratio'], errors='coerce')

# Remove any rows with missing critical data
real_data_clean = real_data_clean.dropna(subset=['mp_kd_ratio', 'mp_legendary_streak', 'experience_level'])

print(f"Clean data: {len(real_data_clean)} records")

# Step 3: Analyze distributions of real data
print("\nAnalyzing real data distributions...")

# Numerical features statistics
kd_mean = real_data_clean['mp_kd_ratio'].mean()
kd_std = real_data_clean['mp_kd_ratio'].std()
kd_min = real_data_clean['mp_kd_ratio'].min()
kd_max = real_data_clean['mp_kd_ratio'].max()

legendary_mean = real_data_clean['mp_legendary_streak'].mean()
legendary_std = real_data_clean['mp_legendary_streak'].std()
legendary_min = real_data_clean['mp_legendary_streak'].min()
legendary_max = real_data_clean['mp_legendary_streak'].max()

level_mean = real_data_clean['experience_level'].mean()
level_std = real_data_clean['experience_level'].std()
level_min = real_data_clean['experience_level'].min()
level_max = real_data_clean['experience_level'].max()

# Categorical features distributions
play_time_dist = real_data_clean['daily_play_time'].value_counts(normalize=True).to_dict()
codm_exp_dist = real_data_clean['codm_experience'].value_counts(normalize=True).to_dict()
preferred_mode_dist = real_data_clean['preferred_mode'].value_counts(normalize=True).to_dict()

print(f"\nK/D Ratio: mean={kd_mean:.2f}, std={kd_std:.2f}, range=[{kd_min:.2f}, {kd_max:.2f}]")
print(f"MP Legendary Streak: mean={legendary_mean:.2f}, std={legendary_std:.2f}, range=[{legendary_min:.0f}, {legendary_max:.0f}]")
print(f"Experience Level: mean={level_mean:.2f}, std={level_std:.2f}, range=[{level_min:.0f}, {level_max:.0f}]")
print(f"\nDaily Play Time Distribution: {play_time_dist}")
print(f"CODM Experience Distribution: {codm_exp_dist}")
print(f"Preferred Mode Distribution: {preferred_mode_dist}")

# Step 4: Generate synthetic data
print(f"\nGenerating 1932 synthetic records...")

def generate_correlated_synthetic_data(n_samples, real_data):
    """
    Generate synthetic data that maintains correlations between features
    """
    synthetic_records = []
    
    # Calculate correlation between K/D and Legendary streak from real data
    correlation = real_data[['mp_kd_ratio', 'mp_legendary_streak']].corr().iloc[0, 1]
    
    for i in range(n_samples):
        # Method 1: Pure statistical generation (60% of data)
        if i < int(n_samples * 0.6):
            # Generate K/D ratio
            kd = np.random.normal(kd_mean, kd_std * 1.1)  # Slightly more variance
            kd = np.clip(kd, 0.5, 5.0)  # Realistic bounds
            
            # Generate correlated legendary streak
            # Higher K/D tends to correlate with more legendary seasons
            legendary_base = (kd - kd_mean) / kd_std * legendary_std * correlation + legendary_mean
            legendary = int(np.clip(legendary_base + np.random.normal(0, legendary_std * 0.5), 0, 25))
            
            # Generate experience level (correlated with legendary streak)
            level_base = 150 + (legendary * 15) + np.random.normal(0, 80)
            level = int(np.clip(level_base, 50, 400))
            
        # Method 2: Perturbation of real records (30% of data)
        elif i < int(n_samples * 0.9):
            base_record = real_data.sample(n=1).iloc[0]
            
            # Add small random variations
            kd = base_record['mp_kd_ratio'] + np.random.normal(0, kd_std * 0.3)
            kd = np.clip(kd, 0.5, 5.0)
            
            legendary = int(base_record['mp_legendary_streak'] + np.random.randint(-2, 3))
            legendary = np.clip(legendary, 0, 25)
            
            level = int(base_record['experience_level'] + np.random.randint(-50, 51))
            level = np.clip(level, 50, 400)
            
        # Method 3: Interpolation between similar records (10% of data)
        else:
            records = real_data.sample(n=2)
            alpha = np.random.random()  # Interpolation weight
            
            kd = records['mp_kd_ratio'].iloc[0] * alpha + records['mp_kd_ratio'].iloc[1] * (1 - alpha)
            kd = np.clip(kd, 0.5, 5.0)
            
            legendary = int(records['mp_legendary_streak'].iloc[0] * alpha + 
                          records['mp_legendary_streak'].iloc[1] * (1 - alpha))
            legendary = max(0, legendary)
            
            level = int(records['experience_level'].iloc[0] * alpha + 
                       records['experience_level'].iloc[1] * (1 - alpha))
            level = np.clip(level, 50, 400)
        
        # Sample categorical features based on real distribution
        daily_play_time = np.random.choice(
            list(play_time_dist.keys()),
            p=list(play_time_dist.values())
        )
        
        codm_experience = np.random.choice(
            list(codm_exp_dist.keys()),
            p=list(codm_exp_dist.values())
        )

        preferred_mode = np.random.choice(
            list(preferred_mode_dist.keys()),
            p=list(preferred_mode_dist.values())
)
        
        # Create synthetic record
        record = {
            'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            'ign': f'Bot_{i+69:04d}',
            'uid': f'synthetic_{i+1}',
            'experience_level': int(level),
            'preferred_mode': preferred_mode,
            'daily_play_time': daily_play_time,
            'codm_experience': codm_experience,
            'mp_legendary_streak': int(legendary),
            'br_legendary_streak': int(legendary * np.random.uniform(0.5, 1.2)),  # BR correlated with MP
            'mp_kd_ratio': round(float(kd), 2),
            'interest_rating': np.random.choice([3, 4, 5], p=[0.1, 0.3, 0.6])
        }
        
        synthetic_records.append(record)

        # Progress indicator
        if (i + 1) % 500 == 0:
            print(f"   Generated {i + 1}/{n_samples} records...")
    
    return pd.DataFrame(synthetic_records)

# Generate synthetic data
synthetic_data = generate_correlated_synthetic_data(1932, real_data_clean)

print(f"Synthetic data generated: {len(synthetic_data)} records")

# Step 5: Combine real and synthetic data
print("\nCombining real and synthetic data...")

# Select only the columns we need from real data
real_data_subset = real_data_clean[['timestamp', 'ign', 'uid', 'experience_level', 'preferred_mode',
    'daily_play_time', 'codm_experience', 'mp_legendary_streak',
    'br_legendary_streak', 'mp_kd_ratio', 'interest_rating']].copy()

# Add source column to track origin
real_data_subset['data_source'] = 'real'
synthetic_data['data_source'] = 'synthetic'

# Combine
combined_data = pd.concat([real_data_subset, synthetic_data], ignore_index=True)

print(f"Total records: {len(combined_data)}")
print(f"Real: {len(real_data_subset)}, Synthetic: {len(synthetic_data)}")

# Step 6: Create the target variable (Performance Score)
print("\nCreating performance score...")

def calculate_performance_score(row):
    """
    Calculate player performance score (0-100) based on weighted features
    
    Weight Distribution:
    - K/D Ratio: 40% (0-40 points)
    - MP Legendary Streak: 25% (0-25 points)
    - Experience Level: 15% (0-15 points)
    - Daily Play Time: 10% (0-10 points)
    - CODM Experience: 10% (0-10 points)
    """
    
    # K/D Ratio score (0-40 points)
    # 0.5 or less = 0 points, 4.0+ = 40 points
    kd_score = min(40, max(0, (row['mp_kd_ratio'] - 0.5) / 3.5 * 40))
    
    # Legendary streak score (0-25 points)
    # Each season = 1 point, capped at 25
    legendary_score = min(25, row['mp_legendary_streak'])
    
    # Experience level score (0-15 points)
    # Level 50 = 0 points, Level 400 = 15 points
    level_score = min(15, max(0, (row['experience_level'] - 50) / 350 * 15))
    
    # Daily play time score (0-10 points)
    play_time_map = {
        'Less than 1 hour': 3,
        '1-2 hours': 6,
        '2-3 hours': 8,
        'More than 3 hours': 10
    }
    play_time_score = play_time_map.get(row['daily_play_time'], 5)
    
    # CODM experience score (0-10 points)
    experience_map = {
        'Less than 1 year': 3,
        '1-2 years': 5,
        '2-3 years': 7,
        'More than 3 years': 10
    }
    experience_score = experience_map.get(row['codm_experience'], 5)
    
    # Calculate total score
    total = kd_score + legendary_score + level_score + play_time_score + experience_score
    
    # Add small random noise (±2 points) for realism
    noise = np.random.normal(0, 1.5)
    final_score = np.clip(total + noise, 0, 100)
    
    return round(final_score, 1)

# Apply performance score calculation
combined_data['performance_score'] = combined_data.apply(calculate_performance_score, axis=1)

# Step 7: Assign player classes
def assign_class(score):
    if score >= 81:
        return 'A'
    elif score >= 61:
        return 'B'
    elif score >= 41:
        return 'C'
    elif score >= 21:
        return 'D'
    else:
        return 'E'

combined_data['player_class'] = combined_data['performance_score'].apply(assign_class)

# Step 8: Display statistics
print("\n" + "="*60)
print("FINAL DATASET STATISTICS")
print("="*60)

print(f"\nTotal records: {len(combined_data)}")
print(f"Real data: {len(combined_data[combined_data['data_source'] == 'real'])}")
print(f"Synthetic data: {len(combined_data[combined_data['data_source'] == 'synthetic'])}")

print("\n--- Performance Score Distribution ---")
print(combined_data['performance_score'].describe())

print("\n--- Player Class Distribution ---")
class_dist = combined_data['player_class'].value_counts().sort_index()
for player_class, count in class_dist.items():
    percentage = count / len(combined_data) * 100
    print(f"   Class {player_class}: {count} players ({percentage:.1f}%)")

print("\nPercentages:")
print(combined_data['player_class'].value_counts(normalize=True).sort_index() * 100)

print("\n--- Feature Statistics ---")
print("\nK/D Ratio:")
print(combined_data['mp_kd_ratio'].describe())
print("\nMP Legendary Streak:")
print(combined_data['mp_legendary_streak'].describe())
print("\nExperience Level:")
print(combined_data['experience_level'].describe())

# Step 9: Save the combined dataset to proper location
output_filename = 'data/processed/zifty_player_data_complete_2000.csv'
combined_data.to_csv(output_filename, index=False)

print(f"\nDataset saved as '{output_filename}'")

# Step 10: Create visualization
print("\nCreating data validation visualizations...")

plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(16, 12))

# 1. Performance Score Distribution
plt.subplot(3, 3, 1)
combined_data[combined_data['data_source'] == 'real']['performance_score'].hist(
    bins=20, alpha=0.7, label='Real', color='#2E86AB', edgecolor='black')
combined_data[combined_data['data_source'] == 'synthetic']['performance_score'].hist(
    bins=20, alpha=0.6, label='Synthetic', color='#A23B72', edgecolor='black')
plt.xlabel('Performance Score', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title('Performance Score Distribution', fontsize=12, fontweight='bold')
plt.legend()

# 2. K/D Ratio Distribution
plt.subplot(3, 3, 2)
combined_data[combined_data['data_source'] == 'real']['mp_kd_ratio'].hist(
    bins=20, alpha=0.7, label='Real', color='#2E86AB', edgecolor='black')
combined_data[combined_data['data_source'] == 'synthetic']['mp_kd_ratio'].hist(
    bins=20, alpha=0.6, label='Synthetic', color='#A23B72', edgecolor='black')
plt.xlabel('K/D Ratio', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title('K/D Ratio Distribution', fontsize=12, fontweight='bold')
plt.legend()

# 3. Legendary Streak Distribution
plt.subplot(3, 3, 3)
combined_data[combined_data['data_source'] == 'real']['mp_legendary_streak'].hist(
    bins=15, alpha=0.7, label='Real', color='#2E86AB', edgecolor='black')
combined_data[combined_data['data_source'] == 'synthetic']['mp_legendary_streak'].hist(
    bins=15, alpha=0.6, label='Synthetic', color='#A23B72', edgecolor='black')
plt.xlabel('MP Legendary Streak', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title('Legendary Streak Distribution', fontsize=12, fontweight='bold')
plt.legend()

# 4. Experience Level Distribution
plt.subplot(3, 3, 4)
combined_data[combined_data['data_source'] == 'real']['experience_level'].hist(
    bins=20, alpha=0.7, label='Real', color='#2E86AB', edgecolor='black')
combined_data[combined_data['data_source'] == 'synthetic']['experience_level'].hist(
    bins=20, alpha=0.6, label='Synthetic', color='#A23B72', edgecolor='black')
plt.xlabel('Experience Level', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title('Experience Level Distribution', fontsize=12, fontweight='bold')
plt.legend()

# 5. Player Class Distribution
plt.subplot(3, 3, 5)
class_counts = combined_data.groupby(['player_class', 'data_source']).size().unstack()
class_counts.plot(kind='bar', ax=plt.gca(), color=['#2E86AB', '#A23B72'], edgecolor='black')
plt.xlabel('Player Class', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.title('Player Class Distribution', fontsize=12, fontweight='bold')
plt.legend(['Real', 'Synthetic'])
plt.xticks(rotation=0)

# 6. K/D vs Legendary Streak Scatter
plt.subplot(3, 3, 6)
real_df = combined_data[combined_data['data_source'] == 'real']
synthetic_count = len(combined_data[combined_data['data_source'] == 'synthetic'])
sample_size = min(200, synthetic_count)  # Sample min of 200 or total available
synthetic_sample = combined_data[combined_data['data_source'] == 'synthetic'].sample(n=sample_size)
plt.scatter(real_df['mp_kd_ratio'], real_df['mp_legendary_streak'], 
           alpha=0.7, label='Real', s=80, color='#2E86AB', edgecolors='black')
plt.scatter(synthetic_sample['mp_kd_ratio'], synthetic_sample['mp_legendary_streak'], 
           alpha=0.4, label='Synthetic', s=40, color='#A23B72')
plt.xlabel('K/D Ratio', fontsize=10)
plt.ylabel('MP Legendary Streak', fontsize=10)
plt.title('K/D vs Legendary Streak Correlation', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# 7. Performance Score by Class (Boxplot)
plt.subplot(3, 3, 7)
combined_data.boxplot(column='performance_score', by='player_class', ax=plt.gca())
plt.xlabel('Player Class', fontsize=10)
plt.ylabel('Performance Score', fontsize=10)
plt.title('Performance Score by Player Class', fontsize=12, fontweight='bold')
plt.suptitle('')

# 8. Daily Play Time Distribution
plt.subplot(3, 3, 8)
play_time_counts = combined_data.groupby(['daily_play_time', 'data_source']).size().unstack(fill_value=0)
play_time_counts.plot(kind='bar', ax=plt.gca(), color=['#2E86AB', '#A23B72'], edgecolor='black')
plt.xlabel('Daily Play Time', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.title('Daily Play Time Distribution', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.legend(['Real', 'Synthetic'])

# 9. CODM Experience Distribution
plt.subplot(3, 3, 9)
exp_counts = combined_data.groupby(['codm_experience', 'data_source']).size().unstack(fill_value=0)
exp_counts.plot(kind='bar', ax=plt.gca(), color=['#2E86AB', '#A23B72'], edgecolor='black')
plt.xlabel('CODM Experience', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.title('CODM Experience Distribution', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.legend(['Real', 'Synthetic'])

plt.tight_layout()
plt.savefig('reports/data_validation_report.png', dpi=300, bbox_inches='tight')
print(f"Saved visualization: reports/data_validation_report.png")

print("\n" + "="*60)
print("DATA GENERATION COMPLETE!")
print("="*60)
print(f"\nGenerated files:")
print(f"  1. {output_filename}")
print(f"  2. reports/data_validation_report.png")
print(f"\nYou can now proceed to model training!")
print("="*60)