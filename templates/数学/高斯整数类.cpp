ll roundiv(ll x, ll y)
{
	return x >= 0 ? (x + y / 2) / y : (x - y / 2) / y;
}
struct Q
{
	ll x, y;
	Q operator~() const { return {x, -y}; }
	ll len2() const { return x * x + y * y; }
	Q operator+(const Q &o) const { return {x + o.x, y + o.y}; }
	Q operator-(const Q &o) const { return {x - o.x, y - o.y}; }
	Q operator*(const Q &o) const { return {x * o.x - y * o.y, x * o.y + y * o.x}; }
	Q operator/(const Q &o) const
	{
		Q t = *this * ~o;
		ll l = o.len2();
		return {roundiv(t.x, l), roundiv(t.y, l)};
	}
	Q operator%(const Q &o) const { return *this - *this / o * o; }
};
Q gcd(Q a, Q b)
{
	if (a.len2() > b.len2()) swap(a, b);
	while (a.len2())
	{
		b = b % a;
		swap(a, b);
	}
	return b;
}


