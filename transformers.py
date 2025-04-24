
def add_interactions(X):
    X = X.copy()
    X['Fuel_Engine'] = X['Fuel Consumption Comb (L/100 km)'] * X['Engine Size(L)']
    X['Fuel_AV'] = X['Fuel Consumption Comb (L/100 km)'] * (X['Transmission'] == 'AV').astype(int)
    X['Class_AV'] = (X['Vehicle Class'] == 'SUV - STANDARD').astype(int) * (X['Transmission'] == 'AV').astype(int)
    X['Cyl_Eng'] = X['Cylinders'] * X['Engine Size(L)']
    return X[['Fuel_Engine', 'Fuel_AV', 'Class_AV', 'Cyl_Eng']]