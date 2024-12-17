
Group for Term Project 7

Members:
1. Amarender Reddy Jakka
   - Student ID: 017536610
   - Email: amarenderreddy.jakka@sjsu.edu

2. Harshini Pothireddy
   - Student ID: 017513548
   - Email: harshini.pothireddy@sjsu.edu

App URL:
	ec2-13-59-128-164.us-east-2.compute.amazonaws.com:5001/

Instructions for Grading:
1. Open the above URL in any web browser.
2. Select a date using the input field and click "Predict."
3. The app will display the following:
   - The highest price, lowest price, and average closing price for the next five business days.
   - A trading strategy (BEARISH, BULLISH, or IDLE) recommended for each of the five days.

UI Preview:
	Predicted Prices Table:
		-Shows the highest, lowest, and average closing prices.
	Recommended Trading Strategy Table:
		-Displays the date and corresponding action (BEARISH, IDLE, or BULLISH).
Example Output:

	Highest Price: $139.34
	Lowest Price: $132.80
	Average Closing Price: $134.24
	Trading Strategy:

	2024-12-14: BEARISH
	2024-12-15: IDLE
	2024-12-16: IDLE
	2024-12-17: IDLE
	2024-12-18: BULLISH

5. Verify the results displayed on the console.

Build Instructions:
- Backend:
   - Framework:  Flask (Python)
   - Dependencies:
     - pandas
     - numpy
     - sklearn
     - flask
     - pickle
- Steps to Run:
      1. Install dependencies:
   
         	pip install -r requirements.txt
         
      2. Run the backend server:
         
       		python app.py
        
      3. Access the app locally at: `http://127.0.0.1:5001/`

- Frontend:
   - Technologies: HTML
   - Integrated with Flask backend.


