namespace edge_coloring
{
	struct Q
	{
		int u, v, id;
	};
	int n, m, ecnt, mx, nn;
	vector<pair<int, int>> eg;
	vector<Q> e;
	vector<int> ans, dfn;
	int id = 0;

	vector<int> compress(int _n, const vector<int> &d, int lim, vector<int> &cd)
	{
		int i;
		vector<int> id(_n, -1), f, s;
		for (i = 0; i < _n; i++) if (d[i])
		{
			id[i] = s.size();
			s.push_back(d[i]);
		}
		f.resize(s.size());
		iota(all(f), 0);
		auto getf = [&](int x) {
			while (f[x] != x) x = f[x] = f[f[x]];
			return x;
		};
		set<pair<int, int>> q;
		for (i = 0; i < s.size(); i++) q.insert({s[i], i});
		while (q.size() >= 2)
		{
			auto it = q.begin();
			if (it->first > lim / 2) break;
			auto jt = next(it);
			if (jt->first > lim / 2) break;
			int u = it->second, v = jt->second;
			q.erase(jt); q.erase(it);
			u = getf(u); v = getf(v);
			f[v] = u;
			s[u] += s[v]; s[v] = 0;
			q.insert({s[u], u});
		}
		cd.clear();
		vector<int> p(s.size(), -1), r(_n, -1);
		for (i = 0; i < _n; i++) if (id[i] != -1)
		{
			int x = getf(id[i]);
			if (p[x] == -1) p[x] = cd.size(), cd.push_back(s[x]);
			r[i] = p[x];
		}
		return r;
	}
	void build()
	{
		int i, j;
		vector<int> dl(n), dr(m);
		mx = 0;
		for (auto [u, v] : eg) ++dl[u], ++dr[v];
		for (int x : dl) mx = max(mx, x);
		for (int x : dr) mx = max(mx, x);
		if (!mx) return;
		vector<int> cl, cr;
		auto ml = compress(n, dl, mx, cl);
		auto mr = compress(m, dr, mx, cr);
		e.clear();
		for (i = 0; i < ecnt; i++)
		{
			auto [u, v] = eg[i];
			e.push_back({ml[u], mr[v], i});
		}
		nn = max(cl.size(), cr.size());
		cl.resize(nn); cr.resize(nn);
		vector<int> sl(nn), sr(nn);
		ll a = 0, b = 0;
		for (i = 0; i < nn; i++)
		{
			sl[i] = mx - cl[i]; sr[i] = mx - cr[i];
			a += sl[i]; b += sr[i];
		}
		assert(a == b);
		for (i = j = 0; i < nn && j < nn;)
		{
			while (i < nn && !sl[i]) ++i;
			while (j < nn && !sr[j]) ++j;
			if (i == nn || j == nn) break;
			int x = min(sl[i], sr[j]);
			while (x--) e.push_back({i, j, -1}), --sl[i], --sr[j];
		}
		assert(e.size() == nn * mx);
	}
	vector<int> matching(const vector<int> &s)
	{
		int i, S = 2 * nn, T = 2 * nn + 1;
		vector<tuple<int, int, ll>> fe;
		fe.reserve(2 * nn + s.size());
		for (i = 0; i < nn; i++) fe.push_back({S, i, 1});
		int mid = fe.size();
		for (int w : s) fe.push_back({e[w].u, nn + e[w].v, 1});
		for (i = 0; i < nn; i++) fe.push_back({nn + i, T, 1});
		auto [w, f] = net::max_flow(T, fe, S, T);
		assert(w == nn);
		vector<int> r;
		for (i = 0; i < s.size(); i++) if (f[mid + i]) r.push_back(s[i]);
		assert(r.size() == nn);
		return r;
	}
	pair<vector<int>, vector<int>> split(const vector<int> &s)
	{
		int i;
		vector<vector<pair<int, int>>> g(2 * nn);
		for (int w : s)
		{
			int u = e[w].u, v = nn + e[w].v;
			g[u].push_back({v, w});
			g[v].push_back({u, w});
		}
		vector<int> ed(e.size());
		vector<int> it(2 * nn), a, b;
		for (i = 0; i < 2 * nn; i++)
		{
			int gi = g[i].size();
			while (it[i] < gi && ed[g[i][it[i]].second]) ++it[i];
			if (it[i] == gi) continue;
			vector<pair<int, int>> st;
			vector<int> r;
			st.push_back({i, -1});
			while (st.size())
			{
				int u = st.back().first;
				int gu = g[u].size();
				while (it[u] < gu && ed[g[u][it[u]].second]) ++it[u];
				if (it[u] == gu)
				{
					int w = st.back().second;
					st.pop_back();
					if (w != -1) r.push_back(w);
				}
				else
				{
					auto [v, w] = g[u][it[u]++];
					if (ed[w]) continue;
					ed[w] = 1;
					st.push_back({v, w});
				}
			}
			assert(r.size() % 2 == 0);
			for (int j = 0; j < r.size(); j++) if (j & 1) b.push_back(r[j]); else a.push_back(r[j]);
		}
		assert(a.size() == b.size());
		return {a, b};
	}
	void dfs(vector<int> s, int k, int c)
	{
		if (!s.size()) return;
		if (k == 1)
		{
			for (int w : s) if (e[w].id != -1) ans[e[w].id] = c;
			return;
		}
		if (k & 1)
		{
			auto mt = matching(s);
			++id;
			for (int x : mt)
			{
				dfn[x] = id;
				if (e[x].id != -1) ans[e[x].id] = c;
			}
			vector<int> t;
			t.reserve(s.size() - mt.size());
			for (int w : s) if (dfn[w] != id) t.push_back(w);
			s.swap(t);
			--k; ++c;
		}
		auto [a, b] = split(s);
		dfs(a, k / 2, c);
		dfs(b, k / 2, c + k / 2);
	}
	vector<int> color(int _n, int _m, const vector<pair<int, int>> &_eg)//[0,n)-[0,m)
	{
		n = _n; m = _m; eg = _eg; ecnt = eg.size();
		id = 0;
		ans.assign(ecnt, -1);
		if (!ecnt) return ans;
		build();
		dfn.assign(e.size(), 0);
		vector<int> p(e.size());
		iota(all(p), 0);
		dfs(p, mx, 0);
		return ans;
	}
}
using edge_coloring::color;
