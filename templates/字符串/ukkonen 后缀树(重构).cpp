struct suffixtree
{
	const static int M = 27;
	struct P
	{
		int v, w;
	};
	struct Q
	{
		int f, t, v;//t=0: n
	};
	vector<Q> edges;
	vector<vector<P>> e;
	vector<array<int, M>> c;
	vector<int> s, fa, dep, siz;
	int n, point, ds, remain, r, edge;
	bool bd;
	suffixtree() :c(2), fa({0, 1}), edges(1), e(2)
	{
		n = remain = r = edge = bd = 0;
		point = ds = 1;
	}
	suffixtree(const string &s) :c(2), fa({0, 1}), edges(1), e(2)
	{
		n = remain = r = edge = bd = 0;
		point = ds = 1;
		reserve(s.size());
		for (auto c : s) insert(c - 'a');
		insert(26);
	}
	void reserve(int len)
	{
		++len;
		s.reserve(len);
		len = len * 2 + 2;
		c.reserve(len);
		fa.reserve(len);
		e.reserve(len);
	}
	inline void add(int a, int b, int cc, int d)
	{
		assert(edges.size());
		c[a][s[cc]] = edges.size();
		edges.push_back({cc, d, b});
	}
	void insert(int ch)//[0,|S|)
	{
		assert(ds == fa.size() - 1 && ds == c.size() - 1 && n == s.size() && ds == e.size() - 1);
		assert(ch >= 0 && ch < M);
		s.push_back(ch);
		int ad = 0;
		++remain;
		while (remain)
		{
			if (!r) edge = n;
			if (int m = c[point][s[edge]]; !m)
			{
				assert(!m);
				fa.push_back(1); c.push_back({ }); e.push_back({ });
				fa[ad] = point;
				add(ad = point, ++ds, edge, -1);
				e[point].push_back({s[edge]});
				//add(point,s[edge]);
			}
			else
			{
				assert(m);
				auto [f, t, v] = edges[m];
				if (t >= 0 && t - f + 1 <= r)
				{
					assert(t != n);
					r -= t - f + 1;
					edge += t - f + 1;
					point = v;
					continue;
				}
				assert(f + r <= n);
				if (s[f + r] == s[n])
				{
					++r;
					fa[ad] = point;
					break;
				}
				fa.push_back(1); c.push_back({ }); e.push_back({ });
				fa.push_back(1); c.push_back({ }); e.push_back({ });
				fa[ad] = ++ds;
				add(ad = ds, v, f + r, t);
				e[ds].push_back({s[n]});
				e[ds].push_back({s[f + r]});
				//add(ds, s[n]); add(ds, s[f + r]);
				++ds; add(ds - 1, ds, n, -1);
				edges[m] = {f, f + r - 1, ds - 1};
			}
			--remain;
			if (r && point == 1)
			{
				--r;
				edge = n - remain + 1;
			}
			else point = fa[point];
		}
		++n;
	}
	void build_edge()
	{
		bd = 1;

		//其余信息
		dep.resize(ds + 1);
		siz.resize(ds + 1);

		int i, j;
		for (i = 1; i <= ds; i++) for (auto &[v, w] : e[i])
		{
			j = c[i][v];
			v = edges[j].v;
			w = (edges[j].t >= 0 ? edges[j].t : n - 1) - edges[j].f + 1;
		}
	}
	void out()
	{
		int i;
		for (i = 1; i <= ds; i++) for (int j : c[i]) if (j)
		{
			auto [f, t, v] = edges[j];
			if (t == -1) t = n - 1;
			cerr << i << ' ' << v << ' ';
			//cerr<<i<<" -> "<<v<<": ";
			for (int k = f; k <= t; k++) cerr << char('a' + s[k]);
			cerr << endl;
		}
	}
	ll ans;
	void dfs(int u)
	{
		assert(bd);
		++ans;
		for (auto [v, w] : e[u])
		{
			//dep[v]=dep[u]+w;
			dfs(v);
			ans += w - 1;
		}
	}
	ll fun()
	{
		ans = 0;
		build_edge();
		dfs(1);
		return ans - n;
	}
};

