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
	for (i = 1; i <= n; i++)
	{
		int mx = 0;
		for (int v : rg[i]) cmax(mx, dis(i, v));
		inc[i] = bit<ll>(mx + 1);
		if (i != DFS::rt)
		{
			mx = 0;
			for (int v : rg[i]) cmax(mx, dis(DFS::f[i], v));
			dec[i] = bit<ll>(mx + 1);
		}
	}
	while (m--)
	{
		int op, u;
		cin >> op >> u; ++u;
		if (op == 0)
		{
			int l, r, x;
			cin >> l >> r >> x;
			auto v = get(u);
			int m = v.size();
			for (i = 0; i < m; i++)
			{
				inc[v[i]].add(l - dis(v[i], u), x);
				inc[v[i]].add(r - dis(v[i], u), -x);
			}
			for (i = 0; i + 1 < m; i++)
			{
				dec[v[i]].add(l - dis(v[i + 1], u), x);
				dec[v[i]].add(r - dis(v[i + 1], u), -x);
			}
		}
		else
		{
			ll res = a[u];
			auto v = get(u);
			int m = v.size();
			for (i = 0; i < m; i++) res += inc[v[i]].sum(dis(v[i], u));
			for (i = 0; i + 1 < m; i++) res -= dec[v[i]].sum(dis(v[i + 1], u));
			cout << res << '\n';
		}
	}
}
