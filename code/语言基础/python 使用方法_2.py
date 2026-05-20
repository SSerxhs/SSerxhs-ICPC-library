class Q:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, o):
        r = Q(self.x + o.x, self.y + o.y)
        return r

    def __sub__(self, o):
        r = Q(self.x - o.x, self.y - o.y)
        return r

    def __mul__(self, o):
        return self.x * o.y - self.y * o.x

    def __lt__(self, o):
        if self.x != o.x:
            return self.x < o.x
        return self.y < o.y


n, m = map(int, input().split())
c = list(map(int, input().split()))
print(*c)
a = Q(0, 0)
b = Q(1, 1)
if a < b - a:
    pass

