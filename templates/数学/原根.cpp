namespace get_root
{
	using ull = unsigned long long;
	using u128 = __uint128_t;
	ull ksm(ull x, ull y, ull p)
	{
		ull r = 1;
		while (y)
		{
			if (y & 1) r = (u128)r * x % p;
			x = (u128)x * x % p; y >>= 1;
		}
		return r;
	}
	template<class T> ll getrt(ull m, T getw)
	{
		assert(m);
		if (m <= 4) return (ll)m - 1;
		ull phi = m;
		auto w = getw(m);
		if (w.size() >= 3 || m % 4 == 0 || w.size() == 2 && w[0] != 2) return -1;
		for (ull x : w) phi = phi / x * (x - 1);
		w = getw(phi);
		for (ull &x : w) x = phi / x;
		for (ull i = 2; i < m; i++) if (gcd(i, m) == 1)
		{
			for (ull x : w) if (ksm(i, x, m) == 1) goto no;
			return i;
		no:;
		}
		return -1;
	}
}
using get_root::getrt;
int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cout << fixed << setprecision(15);
	ull p;
	cin >> p;
	cout << getrt(p, [&](ull m) {
		auto ww = pr::getw(m);
		vector<ull> w;
		for (auto [p, k] : ww) w.push_back(p);
		return w;
	}) << '\n';
}

