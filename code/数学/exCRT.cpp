namespace CRT
{
	pair<ll, ll> exgcd(ll a, ll b, ll c)
	{
		assert(a || b);
		ll d = gcd(a, b);
		if (c % d) return {-1, -1};
		if (!b) return {c / a, 0};
		if (a < 0) a = -a, b = -b, c = -c;
		ll x = 1, x1 = 0, p = a, q = b, k;
		b = abs(b);
		while (b)
		{
			k = a / b;
			x -= k * x1; a -= k * b;
			swap(x, x1);
			swap(a, b);
		}
		b = abs(q / d);
		x = x * (c / d) % b;
		if (x < 0) x += b;
		return {x, (c - p * x) / q};
	}
	struct Q
	{
		ll p, r;//0<=r<p
		Q operator+(const Q &o) const
		{
			if (p == 0 || o.p == 0) return {0, 0};
			auto [x, y] = exgcd(p, -o.p, r - o.r);
			if (x == -1 && y == -1) return {0, 0};
			ll q = lcm(p, o.p);
			return {q, ((r - x * p) % q + q) % q};
		}
	};
}
using CRT::Q;

