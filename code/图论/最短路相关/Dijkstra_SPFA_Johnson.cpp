vector<ll> spfa(const vector<vector<pair<int, ll>>> &e, int s)
{
	int n = e.size(), i;
	assert(n);
	queue<int> q;
	vector<int> len(n), ed(n);
	vector<ll> dis(n, inf);
	q.push(s); dis[s] = 0;
	while (q.size())
	{
		int u = q.front(); q.pop();
		ed[u] = 0;
		for (auto [v, w] : e[u]) if (cmin(dis[v], dis[u] + w))
		{
			len[v] = len[u] + 1;
			if (len[v] > n) return { };
			if (!ed[v])
			{
				ed[v] = 1;
				q.push(v);
			}
		}
	}
	return dis;
}
vector<ll> spfa(const vector<vector<pair<int, ll>>> &e)
{
	int n = e.size(), i;
	assert(n);
	queue<int> q;
	vector<int> len(n), ed(n, 1);
	vector<ll> dis(n);
	for (i = 0; i < n; i++) q.push(i);
	while (q.size())
	{
		int u = q.front(); q.pop();
		ed[u] = 0;
		for (auto [v, w] : e[u]) if (cmin(dis[v], dis[u] + w))
		{
			len[v] = len[u] + 1;
			if (len[v] > n) return { };
			if (!ed[v])
			{
				ed[v] = 1;
				q.push(v);
			}
		}
	}
	return dis;
}
vector<ll> dijk(const vector<vector<pair<int, ll>>> &e, int s)
{
	int n = e.size();
	using pa = pair<ll, int>;
	vector<ll> d(n, inf);
	vector<int> ed(n);
	priority_queue<pa, vector<pa>, greater<pa>> q;
	d[s] = 0; q.push({0, s});
	while (q.size())
	{
		int u = q.top().second; q.pop();
		ed[u] = 1;
		for (auto [v, w] : e[u]) if (cmin(d[v], d[u] + w)) q.push({d[v], v});
		while (q.size() && ed[q.top().second]) q.pop();
	}
	return d;
}
vector<vector<ll>> dijk(const vector<vector<pair<int, ll>>> &e)
{
	vector<vector<ll>> r;
	for (int i = 0; i < e.size(); i++) r.push_back(dijk(e, i));
	return r;
}
vector<vector<ll>> john(vector<vector<pair<int, ll>>> e)
{
	int n = e.size(), i, j;
	assert(n);
	auto h = spfa(e);
	if (!h.size()) return { };
	for (i = 0; i < n; i++) for (auto &[v, w] : e[i]) w += h[i] - h[v];
	auto r = dijk(e);
	for (i = 0; i < n; i++) for (j = 0; j < n; j++) if (r[i][j] != inf) r[i][j] -= h[i] - h[j];
	return r;
}
