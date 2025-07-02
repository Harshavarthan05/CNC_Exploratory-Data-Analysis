import matplotlib.pyplot as plt

plt.scatter(df['SquareFeet'], df['Price'], color='blue')  # lowercase 'scatter'
plt.xlabel('Square Feet')  # fixed typo
plt.ylabel('Price')
plt.title('Square Feet vs Price')
plt.show()
