# Password Strength Classifier: 

The passwords used in our analysis are from webhost leak that is available online. How did we figure out which passwords were stronger and which were weaker? Well, there is a tool called PARS by Georgia Tech university which have all the commercial password meters integrated into it. All I did was give that tool all the passwords and it gave me new files for each commercial password strength meter. The files contained the passwords with one more column i.e their strength based on the commercial password strength meters.
The commercial password strength algorithms I used are of Twitter, Microsoft and battle. How is this algorithm different from these strength meters? First of all, it is entirely based on machine learning rather than on rules. Secondly, I only kept those passwords that were flagged weak, medium and strong by all three strength meters. This means that all the passwords were indeed either weak, medium or strong.

I had a total of 3 million passwords but after taking the intersection of all classifications of commercial meters, I was left with 0.7 million passwords. The reduction was because of the fact that I only used passwords that were flagged in a particular category by all three algorithms.

Password - 670k unique values for password collected online

Strength - three values(0 , 1 , 2) i.e. 0 for weak, 1 for medium, 2 for strong.
Strength of the password based on rules(such as containing digits, special symbols , etc.)
