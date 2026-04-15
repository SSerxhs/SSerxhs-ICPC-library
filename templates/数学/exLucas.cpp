struct binom
{
	using pa = pair<ui, ui>;
	ull p;
	vector<pa> a;
	vector<vector<ui>> b, ib, pw, pb;
	vector<ui> ph, xs;
	ull ksm(ull x, ll y, ull p)
	{
		ull r = 1;
		while (y)
		{
			if (y & 1) r = r * x % p;
			x = x * x % p;
			y >>= 1;
		}
		return r;
	}
	ull f(ll n, ll m, ll nm, int i)
	{
		auto [qi, pi] = a[i];
		ull r = 1;
		ll c1 = 0, c2 = 0;
		while (n)
		{
			r = r * b[i][n % pi] % pi * ib[i][m % pi] % pi * ib[i][nm % pi] % pi;
			c2 += n / pi - m / pi - nm / pi;
			n /= qi, m /= qi, nm /= qi;
			c1 += n - m - nm;
		}
		return 1llu * pw[i][min<int>(c1, pw[i].size() - 1)] * pb[i][c2 % ph[i]] % pi * r % pi;
	}
	ull operator()(ll n, ll m)
	{
		if (m < 0 || n < m) return 0;
		ull r = 0;
		for (int i = 0; i < a.size(); i++) r = (r + xs[i] * f(n, m, n - m, i)) % p;
		return r;
	}
	binom(ull p) :p(p)
	{
		int i, j;
		ull x = p, y, z;
		for (i = 2; i * i <= x; i++) if (x % i == 0)
		{
			z = x; x /= i;
			while (1)
			{
				y = x / i;
				if (i * y == x) x = y; else break;
			}
			a.push_back({i, z / x});
		}
		if (x > 1) a.push_back({x, x});
		int n = a.size();
		b = ib = pw = pb = vector<vector<ui>>(n);
		ph = xs = vector<ui>(n);
		for (i = 0; i < n; i++)
		{
			auto [qi, pi] = a[i];
			ph[i] = pi / qi * (qi - 1);
			xs[i] = ksm(p / pi, ph[i] - 1, p) * (p / pi) % p;
		}
		for (i = 0; i < n; i++)
		{
			auto [qi, pi] = a[i];
			b[i] = ib[i] = vector<ui>(pi, 1);
			for (j = 1; j < pi; j++) b[i][j] = 1llu * b[i][j - 1] * (j % qi == 0 ? 1 : j) % pi;
			ib[i][pi - 1] = ksm(b[i][pi - 1], ph[i] - 1, pi);
			for (j = pi - 1; j; j--) ib[i][j - 1] = 1llu * ib[i][j] * (j % qi == 0 ? 1 : j) % pi;
			pw[i] = {1};
			while (pw[i].back()) pw[i].push_back(1llu * pw[i].back() * qi % pi);
			pb[i].resize(ph[i], 1);
			for (j = 1; j < ph[i]; j++) pb[i][j] = 1llu * pb[i][j - 1] * b[i][pi - 1] % pi;
		}
	}
};
int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cout << fixed << setprecision(15);
	int T, p; cin >> T >> p;
	binom s(p);
	while (T--)
	{
		ll n, m;
		cin >> n >> m;
		cout << s(n, m) << '\n';
	}
}

