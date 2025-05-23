# EXNO-5-DS-DATA VISUALIZATION USING MATPLOT LIBRARY

# Aim:
  To Perform Data Visualization using matplot python library for the given datas.

# EXPLANATION:
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

# Algorithm:
STEP 1:Include the necessary Library.

STEP 2:Read the given Data.

STEP 3:Apply data visualization techniques to identify the patterns of the data.

STEP 4:Apply the various data visualization tools wherever necessary.

STEP 5:Include Necessary parameters in each functions.

# Coding and Output:
```
from google.colab import drive
drive.mount('/content/drive')
```
## Data visualization using Matplotlib
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [3, 6, 2, 7, 1]

# CREATE A LINE GRAPH FOR X AND Y AND LABEL X axis and Y Axis and create a legend
plt.plot(x, y, label='Line 1')  # Plotting the line
plt.xlabel('X Axis')  # Labeling the X axis
plt.ylabel('Y Axis')  # Labeling the Y axis
plt.legend()  # Displaying the legend
plt.show()
```
![440711082-703e53ce-37cc-4b8b-b622-082b9b0f4237](https://github.com/user-attachments/assets/427b8f84-91f6-45d3-8f61-8f4257585286)
```
# line 1 points
x1 = [1,2,3]
y1 = [2,4,1]
x2 = [1,2,3]
y2 = [4,1,3]

# plot line 1 and line 2 points in same graph and include the necessary parameters
plt.plot(x1, y1, label='Line 1')  # Plotting Line 1
plt.plot(x2, y2, label='Line 2')  # Plotting Line 2
plt.xlabel('X Axis')  # Labeling the X axis
plt.ylabel('Y Axis')  # Labeling the Y axis
plt.legend()  # Displaying the legend
plt.show()
```
![440711316-9ca9bc9d-9c47-42ca-900e-11baa2581e11](https://github.com/user-attachments/assets/f0087c9c-b25c-4f5d-9969-8c2c680d5288)
```
# plot the points in the above graph
plt.scatter(x1, y1, color='blue', label='Points Line 1')  # Plotting points for Line 1
plt.scatter(x2, y2, color='red', label='Points Line 2')  # Plotting points for Line 2
plt.xlabel('X Axis')  # Labeling the X axis
plt.ylabel('Y Axis')  # Labeling the Y axis
plt.legend()  # Displaying the legend
plt.show()
```
![440711552-b90b4cb1-d0d2-43f7-9015-6ca7d51cb91f](https://github.com/user-attachments/assets/80dc87ac-6cc3-4991-b675-5c61633a64d5)
```
# Creating some random data
x = [1, 2, 3, 4, 5]
y1 = [10, 12, 14, 16, 18]
y2 = [5, 7, 9, 11, 13]
y3 = [2, 4, 6, 8, 10]


x_values = [0,1,2,3,4,5]
y_values = [0,1,4,9,16,25]

# implement line graph using fill between option
plt.plot(x_values, y_values, label='Line with Fill')
plt.fill_between(x_values, y_values, color='lightblue', alpha=0.5)  # Filling the area under the curve
plt.xlabel('X Axis')  # Labeling the X axis
plt.ylabel('Y Axis')  # Labeling the Y axis
plt.legend()  # Displaying the legend
plt.show()

from scipy.interpolate import make_interp_spline
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 4, 5, 7, 8, 8, 9, 10, 11, 12])
```
![440711813-9fda629c-aa5f-4512-8805-676a8b723d79](https://github.com/user-attachments/assets/50e5fba7-087e-4c87-be35-1e06a4bf0d63)
```
# interpolate data using cubic spline
spl = make_interp_spline(x, y, k=3)  # Cubic spline interpolation
x_new = np.linspace(x.min(), x.max(), 500)  # Create new x values for smooth curve
y_new = spl(x_new)  # Get interpolated y values

# Plotting the original and interpolated curve
plt.plot(x, y, 'o', label='Original Data')  # Plotting original data points
plt.plot(x_new, y_new, label='Cubic Spline Interpolation')  # Plotting the smoothed curve
plt.xlabel('X Axis')  # Labeling the X axis
plt.ylabel('Y Axis')  # Labeling the Y axis
plt.legend()  # Displaying the legend
plt.show()
```
![440711960-280de9fa-92c9-4875-9930-0279173230fe](https://github.com/user-attachments/assets/e5afc803-fe54-4295-af1c-49b728153dad)

## Bar Graph
```
values = [5, 6, 3, 7, 2]
names  = ["A", "B", "C", "D", "E"]

# Create bar graph using the above two variables and include the necessary parameters
plt.bar(names, values, color='skyblue')  # Creating the bar graph with color
plt.xlabel('Categories')  # Labeling the X axis
plt.ylabel('Values')  # Labeling the Y axis
plt.title('Bar Graph')  # Adding a title
plt.show()

c1 = ['red', 'green']
c2 = ['b', 'g']  # we can use this for color
```
![441084976-48c671c6-69a3-431b-b6ab-770e55eb6053](https://github.com/user-attachments/assets/2dbfac39-b6ef-4c96-8180-0d497175e203)
```
# plot a bar chart
plt.bar(names, values, color=c1)  # Bar chart with specific colors
plt.xlabel('Categories')  # Labeling the X axis
plt.ylabel('Values')  # Labeling the Y axis
plt.title('Colored Bar Graph')  # Adding a title
plt.show()
```
![441085097-f7e51220-45c3-43d2-876f-159a859fd248](https://github.com/user-attachments/assets/e1134bf9-f8c5-44b5-b2d9-0ab0cc21898b)

```
df = sns.load_dataset("tips")
df
```
![441085213-c4a1c60e-e102-418b-ad25-23d71a762bc7](https://github.com/user-attachments/assets/432f274d-fc48-4c3e-9315-1010941b7128)
```
plt.figure(figsize=(8, 6))
avg_total_bill = df.groupby('day')['total_bill'].mean()  # Average total bill by day
avg_tip = df.groupby('day')['tip'].mean()  # Average tip by day
p1 = plt.bar(avg_total_bill.index, avg_total_bill, label='Total Bill')
p2 = plt.bar(avg_tip.index, avg_tip, bottom=avg_total_bill, label='Tip')  # Stacked bar chart
# Set the labels and title
plt.xlabel('Day of the Week')
plt.ylabel('Amount')
plt.title('Average Total Bill and Tip by Day')
plt.legend()
plt.show()
```
![441085331-65775438-6b0b-4382-adc6-41e7723c1392](https://github.com/user-attachments/assets/082ee518-95ba-442b-b4c5-ad31f9146046)

## Scatter plot
```
x_values = [0,1,2,3,4,5]
y_values = [0,1,4,9,16,25]

# CREATE A SCATTER PLOT FOR X_VALUES AND Y_VALUES
plt.scatter(x_values, y_values, color='blue', label='Data Points')  # Scatter plot
plt.xlabel('X Axis')  # Labeling the X axis
plt.ylabel('Y Axis')  # Labeling the Y axis
plt.title('Scatter Plot')  # Adding a title
plt.legend()  # Displaying the legend
plt.show()
```
![441085614-769ec04c-4817-4979-bd6a-6798bf0bccd3](https://github.com/user-attachments/assets/127dfed1-138d-4310-af71-7a7bac171e77)
```
# x-axis values
x = [1,2,3,4,5,6,7,8,9,10]
# y-axis values
y = [2,4,5,7,6,8,9,11,12,12]

# plot the above points x and y in scatter plot Using the parameters label= "stars", color="green", marker="*", s=30
plt.scatter(x, y, label="stars", color="green", marker="*", s=30)  # Scatter plot with specific styling
plt.xlabel('X Axis')  # Labeling the X axis
plt.ylabel('Y Axis')  # Labeling the Y axis
plt.title('Scatter Plot with Custom Markers')  # Adding a title
plt.legend()  # Displaying the legend
plt.show()
```
![441085785-e5cf6ed3-148e-4d0d-bb30-5bd187eded00](https://github.com/user-attachments/assets/6e6cf8a8-f483-4d6e-a115-0ce1fb12e9c4)
## Pie-chart
```
# defining labels
activities = ['eat', 'sleep', 'work', 'play']
# portion covered by each label
slices = [3, 7, 8, 6]
# color for each label
colors = ['r', 'y', 'g', 'b']

# plot the pie chart using above parameters
plt.pie(slices, labels=activities, colors=colors, startangle=90, autopct='%1.1f%%')  # Pie chart
plt.title('Pie Chart of Activities')  # Adding a title
plt.show()
```
![441086021-96c0cf19-416a-4e25-9508-d60c48a8cc71](https://github.com/user-attachments/assets/c98288c4-9b8a-4089-abe6-c955170586b1)



# Result:
Thus, successfully Performed Data Visualization using matplot python library for the given datas.
