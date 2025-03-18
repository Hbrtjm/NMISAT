class Quadratic:
    def __init__(self,a,b,c):
        self.a = a
        self.b = b
        self.c = c

    def evaluate_simple(self,x):
        return x**2 * self.a + self.b * x + self.c
    
    def evaluate_horner(self,x):
        return self.c + x * ( self.b + x * self.a ) 
    
    def delta(self):
        return self.b ** 2 - 4*self.a*self.c
    

def main():
    a = 1.22
    b = 3.34
    c = 2.28
    q = Quadratic(a,b,c)
    delta = q.delta()
    print(f"The value of delta is {delta}")
    real_delta = 0.0292
    print(f"The real value of delta is {real_delta}")
    print(f"Relatice difference  between the real_delta and delta {abs(real_delta-delta)/real_delta}")

if __name__ == "__main__":
    main()