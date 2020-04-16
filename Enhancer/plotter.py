# PLOTTING GRAPHS: (NOT FOR BAR CHARTS):


X = [1, 2] # Fake Fraction
Y = [1, 2] # Trend Fraction




fig, ax = plt.subplots()
ax.plot(X, Y, label='Trend')

# Add some text for labels, title and custom x-axis labels, etc.
ax.set_xlabel('Fake Fraction')
ax.set_ylabel('Trend Fraction')
ax.set_title('Fake Fraction v/s Trend Fraction')
ax.legend()

fig.tight_layout()

plt.show()
fig.savefig('Enhancer_Graph_1.png', dpi = 400)