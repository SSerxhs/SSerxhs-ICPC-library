__uint128_t brt = ((__uint128_t)1 << 64) / mod;
for (int i = 1; i <= n; i++)
{
	ans *= i;
	ans = ans - mod * (brt * ans >> 64);
	while (ans >= mod) ans -= mod;//可以替换为 if，但据说会变慢。如果循环展开则需要替换
}
struct barret
{
	ll p, m; //p 表示上面的模数, m 为取模参数
	int c = 0;
	inline void init(ll t) {
		c = 48 + log2(t), p = t;
		m = (ll((ulll(1) << c) / t));
	}
	friend inline ll operator % (ll n, const barret &d)
	{ // get n % d
		ll r = n - ((ulll(n) * d.m) >> d.c) * d.p;
		while (r >= d.p) r -= d.p;
		return r;
	}
} modp;
