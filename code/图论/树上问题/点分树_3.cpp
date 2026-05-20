int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cout << fixed << setprecision(15);
	int n, m, i;
	cin >> n >> m;
	vector<int> a(n + 1);
	for (i = 1; i <= n; i++) cin >> a[i];
	DFS::init(n);
	for (i = 1; i < n; i++)
	{
		int u, v;
		cin >> u >> v;
		++u; ++v;
		e[u].push_back(v);
		e[v].push_back(u);
	}
	DFS::fun();
	vector<bit<ll>> inc(n + 1), dec(n + 1);
	vector<ll> tmp(n + 1);
	for (i = 1; i <= n; i++)
	{
		int mx = 0;
		for (int v : rg[i])
		{
			int d = dis(i, v);
			cmax(mx, d);
			tmp[d] += a[v];
		}
		inc[i] = bit<ll>(mx + 1, tmp.data());
		fill_n(tmp.begin(), mx + 1, 0);
		if (i != DFS::rt)
		{
			mx = 0;
			for (int v : rg[i])
			{
				int d = dis(DFS::f[i], v);
				cmax(mx, d);
				tmp[d] += a[v];
			}
			dec[i] = bit<ll>(mx + 1, tmp.data());
			fill_n(tmp.begin(), mx + 1, 0);
		}
	}
	while (m--)
	{
		int op, u;
		cin >> op >> u; ++u;
		if (op == 0)
		{
			int x;
			cin >> x;
			auto v = get(u);
			int m = v.size();
			for (i = 0; i < m; i++) inc[v[i]].add(dis(v[i], u), x);
			for (i = 0; i + 1 < m; i++) dec[v[i]].add(dis(v[i + 1], u), x);
		}
		else
		{
			int l, r;
			cin >> l >> r;
			--r;
			ll res = 0;
			auto v = get(u);
			int m = v.size();
			for (i = 0; i < m; i++) res += inc[v[i]].sum(l - dis(v[i], u), r - dis(v[i], u));
			for (i = 0; i + 1 < m; i++) res -= dec[v[i]].sum(l - dis(v[i + 1], u), r - dis(v[i + 1], u));
			cout << res << '\n';
		}
	}
}
