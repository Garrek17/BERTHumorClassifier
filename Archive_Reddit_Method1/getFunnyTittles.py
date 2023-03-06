import csv

#Data file headers
#image_id,unixtime,rawtime,title,total_votes,reddit_id,number_of_upvotes,
#subreddit,number_of_downvotes,localtime,score,number_of_comments,username,,

csv_reader = csv.reader(open("Data/submissions.csv", "r"))
# csv_writer = csv.writer(open("Data/submissions_funny.csv", "w"))

# csv_writer.writerow(next(csv_reader))
count = 0
for row in csv_reader:
    if row[7] == "funny":
        print(int(row[4])+1)
        # csv_writer.writerow(row)
        count+=1
print(count)
