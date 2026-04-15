struct AC
{
	const static int N = 3e6 + 2, M = 26;
	int c[N][M], sz[N], pos[N], f[N], app[N];//sz 维护有多少个以当前字符串为前缀的字符串。
	int cnt = 0, id = 0;
	vector<int> q;
	void insert(string s)
	{
		int u = 0;
		++sz[u];
		for (char ch : s)
		{
			assert(ch >= 0 && ch < M);
			int &v = c[u][ch];
			if (!v) v = ++cnt;
			u = v;
			++sz[u];
		}
		pos[id++] = u;
		//此时 u 是字符串结束位置。你可以在此存储结点信息。
	}
	vector<int> match(string s)//返回答案。复杂度 O(结点数)
	{
		int u = 0, i;
		for (char ch : s)
		{
			assert(ch >= 0 && ch < M);
			u = c[u][ch];
			++app[u];
		}
		for (int u : q) app[f[u]] += app[u];
		vector<int> r(id);
		for (i = 0; i < id; i++) r[i] = app[pos[i]];
		memset(app, 0, (cnt + 1) * sizeof app[0]);
		return r;
	}
	void clear()
	{
		memset(c, 0, (cnt + 1) * sizeof c[0]);
		memset(f, 0, (cnt + 1) * sizeof f[0]);
		memset(sz, 0, (cnt + 1) * sizeof sz[0]);
		cnt = id = 0;
	}
	void build()
	{
		q.clear();
		int ql = 0;
		for (int i = 0; i < M; i++) if (c[0][i]) q.push_back(c[0][i]);
		while (ql < q.size())
		{
			int u = q[ql++];
			for (int i = 0; i < M; i++) if (c[u][i])
			{
				q.push_back(c[u][i]);
				f[c[u][i]] = c[f[u]][i];
			}
			else c[u][i] = c[f[u]][i];
		}
		reverse(all(q));
	}
} s;
int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	int n, i;
	cin >> n;
	while (n--)
	{
		string t;
		cin >> t;
		for (char &c : t) c -= 'a';
		s.insert(t);
	}
	s.build();
	string t;
	cin >> t;
	for (char &c : t) c -= 'a';
	auto res = s.match(t);
	for (int x : res) cout << x << '\n';
}
