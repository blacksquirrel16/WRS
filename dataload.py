
import pandas as pd
import matplotlib.pyplot as plt
def load_and_clean_data(train_path, test_path):
    # Load the datasets
    train_df = pd.read_csv(train_path, sep='\t')
    test_df = pd.read_csv(test_path, sep='\t')

    # Remove missing ratings
    train_df = train_df.dropna(subset=['rating'])
    test_df = test_df.dropna(subset=['rating'])

    # Remove duplicates# Remove duplicates, keeping the most recent rating
    train_df = train_df.sort_values(by=['user_id', 'item_id', 'timestamp']).drop_duplicates(subset=['user_id', 'item_id'], keep='last')
    test_df = test_df.sort_values(by=['user_id', 'item_id', 'timestamp']).drop_duplicates(subset=['user_id', 'item_id'], keep='last')

    # Ensure all users in test are in training set
    common_users = set(train_df['user_id']).intersection(set(test_df['user_id']))
    test_df = test_df[test_df['user_id'].isin(common_users)]

    return train_df, test_df

def statistics(train_df):
    # Distribution of ratings per user
    ratings_per_user = train_df.groupby('user_id')['rating'].count()

    # Distribution of ratings per item
    ratings_per_item = train_df.groupby('item_id')['rating'].count()

    # Top 5 most popular items
    top_5_items = ratings_per_item.sort_values(ascending=False).head(5)

    # Displaying the statistics
    print("Ratings per user:")
    print(ratings_per_user.describe())

    print("\nRatings per item:")
    print(ratings_per_item.describe())

    print("\nTop 5 most popular items:")
    print(top_5_items)
    plt.figure(figsize=(10, 6))
    plt.hist(ratings_per_user, bins=20, color='blue', edgecolor='black')
    plt.title('Distribution of Ratings per User')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    # Plotting the distribution of ratings per item
    plt.figure(figsize=(10, 6))
    plt.hist(ratings_per_item, bins=20, color='green', edgecolor='black')
    plt.title('Distribution of Ratings per Item')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    item_downloads = train_df.groupby('item_id').size().reset_index(name='download_count')

    # Sort items by number of downloads in descending order
    item_downloads = item_downloads.sort_values(by='download_count', ascending=False)

    # Plot the long-tail distribution for items
    plt.figure(figsize=(12, 6))
    plt.plot(item_downloads['download_count'].values)
    plt.xlabel('Items (sorted by number of downloads)')
    plt.ylabel('Number of Downloads')
    plt.title('Long-Tail Distribution of Item Downloads')
    plt.grid(True)
    plt.show()

    # Group by user_id to get the number of downloads per user
    user_downloads = train_df.groupby('user_id').size().reset_index(name='download_count')

    # Sort users by number of downloads in descending order
    user_downloads = user_downloads.sort_values(by='download_count', ascending=False)

    # Plot the long-tail distribution for users
    plt.figure(figsize=(12, 6))
    plt.plot(user_downloads['download_count'].values)
    plt.xlabel('Users (sorted by number of downloads)')
    plt.ylabel('Number of Downloads')
    plt.title('Long-Tail Distribution of User Downloads')
    plt.grid(True)
    plt.show()



