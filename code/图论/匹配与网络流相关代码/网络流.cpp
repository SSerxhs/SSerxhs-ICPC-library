namespace net
{
	const int N = 4e5 + 50;//number of nodes
	namespace flow
	{
		const ll inf = 4e18;
		struct Q
		{
			int v;
			ll w;
			int id;
		};
		vector<Q> e[N];
		vector<Q>::iterator fir[N];
		int fc[N], q[N];
		int n, s, t;
		int bfs()
		{
			for (int i = 0; i < n; i++)
			{
				fir[i] = e[i].begin();
				fc[i] = 0;
			}
			int p1 = 0, p2 = 0, u;
			fc[s] = 1; q[0] = s;
			while (p1 <= p2)
			{
				int u = q[p1++];
				for (auto [v, w, id] : e[u]) if (w && !fc[v])
				{
					q[++p2] = v;
					fc[v] = fc[u] + 1;
				}
			}
			return fc[t];
		}
		ll dfs(int u, ll maxf)
		{
			if (u == t) return maxf;
			ll j = 0, k;
			for (auto &it = fir[u]; it != e[u].end(); ++it)
			{
				auto &[v, w, id] = *it;
				if (w && fc[v] == fc[u] + 1 && (k = dfs(v, min(maxf - j, w))))
				{
					j += k;
					w -= k;
					e[v][id].w += k;
					if (j == maxf) return j;
				}
			}
			fc[u] = 0;
			return j;
		}
		pair<ll, vector<ll>> max_flow(int _n, const vector<tuple<int, int, ll>> &edges, int _s, int _t)//[0,n]
		{
			s = _s; t = _t; n = _n + 1;
			for (int i = 0; i < n; i++) e[i].clear();
			for (auto [u, v, w] : edges)
			{
				e[u].push_back({v, w, (int)e[v].size()});
				e[v].push_back({u, 0, (int)e[u].size() - 1});
			}
			ll r = 0;
			while (bfs()) r += dfs(s, inf);
			vector<ll> ans, id(n);
			for (auto [u, v, w] : edges)
			{
				++id[u];
				ans.push_back(e[v][id[v]++].w);
			}
			return {r, ans};
		}
	}
	using flow::max_flow, flow::fc;
	namespace match
	{
		int lk[N], kl[N], ed[N];
		vector<int> e[N];
		int max_match(int n, int m, const vector<pair<int, int>> &edges)//lk[[0,n]]->[0,m]
		{
			++n; ++m;
			int s = n + m, t = n + m + 1, i;
			vector<tuple<int, int, ll>> eg;
			eg.reserve(n + m + edges.size());
			for (i = 0; i < n; i++) eg.push_back({s, i, 1});
			for (i = 0; i < m; i++) eg.push_back({i + n, t, 1});
			for (auto [u, v] : edges) eg.push_back({u, v + n, 1});
			int r = max_flow(t, eg, s, t).first;
			fill_n(lk, n, -1); fill_n(kl, m, -1);
			for (i = 0; i < n; i++) for (auto [v, w, id] : flow::e[i]) if (v < s && !w)
			{
				lk[i] = v - n;
				kl[v - n] = i;
				break;
			}
			return r;
		}
		void dfs(int u)
		{
			for (int v : e[u]) if (!ed[v]) ed[v] = 1, dfs(kl[v]);
		}
		pair<vector<int>, vector<int>> min_cover(int n, int m, const vector<pair<int, int>> &edges)//[0,n]-[0,m]
		{
			max_match(n++, m++, edges);
			fill_n(ed, m, 0);
			int i;
			for (i = 0; i < n; i++) e[i].clear();
			for (auto [u, v] : edges) e[u].push_back(v);
			for (i = 0; i < n; i++) if (lk[i] == -1) dfs(i);
			vector<int> r[2];
			for (i = 0; i < m; i++) if (kl[i] != -1)
			{
				if (ed[i]) r[1].push_back(i); else r[0].push_back(kl[i]);
			}
			sort(all(r[0]));
			return {r[0], r[1]};
		}
	}
	using match::max_match, match::min_cover, match::lk, match::kl;
	namespace cost_flow
	{
		const ll inf = 3e18;
		struct Q
		{
			int v;
			ll w, c;
			int id;
		};
		vector<Q> e[N];
		ll dis[N];
		int pre[N], pid[N], ipd[N];
		bool ed[N];
		int n, s, t;
		pair<ll, lll> spfa()
		{
			queue<int> q;
			fill_n(dis, n, inf);
			memset(ed, 0, n * sizeof ed[0]);
			q.push(s); dis[s] = 0;
			while (q.size())
			{
				int u = q.front(); q.pop(); ed[u] = 0;
				for (auto [v, w, c, id] : e[u]) if (w && cmin(dis[v], dis[u] + c))
				{
					pre[v] = u;
					pid[v] = e[v][id].id;
					ipd[v] = id;
					if (!ed[v]) q.push(v), ed[v] = 1;
				}
			}
			if (dis[t] == inf) return {0, 0};
			ll mw = 9e18;
			for (int i = t; i != s; i = pre[i]) mw = min(mw, e[pre[i]][pid[i]].w);
			for (int i = t; i != s; i = pre[i]) e[pre[i]][pid[i]].w -= mw, e[i][ipd[i]].w += mw;
			return {mw, (lll)mw * dis[t]};
		}
		tuple<ll, lll, vector<ll>> mcmf_spfa(int _n, const vector<tuple<int, int, ll, ll>> &edges, int _s, int _t)//[0,n]
		{
			s = _s; t = _t; n = _n + 1;
			for (int i = 0; i < n; i++) e[i].clear();
			for (auto [u, v, w, c] : edges)
			{
				e[u].push_back({v, w, c, (int)e[v].size()});
				e[v].push_back({u, 0, -c, (int)e[u].size() - 1});
			}
			ll w = 0, dw;
			lll c = 0, dc;
			do
			{
				tie(dw, dc) = spfa();
				w += dw, c += dc;
			} while (dw);
			vector<ll> ans, id(n);
			for (auto [u, v, w, c] : edges)
			{
				++id[u];
				ans.push_back(e[v][id[v]++].w);
			}
			return {w, c, ans};
		}
		pair<ll, ll> spfa_loop()
		{
			vector<ll> d(n);
			vector<int> ed(n, 1), cnt(n), pre(n), pid(n);
			queue<int> q;
			int i;
			for (i = 0; i < n; i++) q.push(i);
			while (q.size())
			{
				int u = q.front(); q.pop(); ed[u] = 0;
				for (auto [v, w, c, id] : e[u]) if (w && cmin(d[v], d[u] + c))
				{
					pre[v] = u;
					pid[v] = id;
					if (d[v]<-inf * 2 || (cnt[v] = cnt[u] + 1) > n)
					{
						ll tw = 0, tc = 0;
						for (u = v; ed[u] <= 1; u = pre[u]) ed[u] = 2;
						for (; ed[u] == 2; u = v)
						{
							v = pre[u];
							++e[u][pid[u]].w;
							--e[v][e[u][pid[u]].id].w;
							if (e[v][e[u][pid[u]].id].c != -inf) tc += e[v][e[u][pid[u]].id].c;
							else tw = 1;
							ed[u] = 3;
						}
						return {tw, tc};
					}
					if (!ed[v]) ed[v] = 1, q.push(v);
				}
			}
			return {0, 0};
		}
		tuple<ll, lll, vector<ll>> mcmf_spfa_scaling(int _n, vector<tuple<int, int, ll, ll>> edges, int _s, int _t)//[0,n]
		{
			s = _s; t = _t; n = _n + 1;
			for (int i = 0; i < n; i++) e[i].clear();
			ll tot = 1;
			for (auto [u, v, w, c] : edges) tot += w;
			edges.push_back({t, s, tot, -inf});//最后一项 mcmf:-inf, mcf:0
			for (auto [u, v, w, c] : edges)
			{
				e[u].push_back({v, 0, c, (int)e[v].size()});
				e[v].push_back({u, 0, -c, (int)e[u].size() - 1});
			}
			ll w = 0, dw;
			lll c = 0, dc;
			for (int g = __lg(tot); g >= 0; g--)
			{
				vector<int> id(n);
				for (auto [u, v, w, c] : edges)
				{
					(e[u][id[u]++].w *= 2) += w >> g & 1;
					(e[v][id[v]++].w *= 2);
				}
				w *= 2, c *= 2;
				do
				{
					tie(dw, dc) = spfa_loop();
					w += dw, c += dc;
				} while (dw || dc < 0);
			}
			vector<ll> ans, id(n);
			edges.pop_back();
			e[s].pop_back(), e[t].pop_back();
			for (auto [u, v, w, c] : edges)
			{
				++id[u];
				ans.push_back(e[v][id[v]++].w);
			}
			return {w, c, ans};
		}
		tuple<ll, lll, vector<ll>> mcmf_dijk(int _n, const vector<tuple<int, int, ll, ll>> &edges, int _s, int _t)//[0,n]
		{
			s = _s; t = _t; n = _n + 1;
			for (int i = 0; i < n; i++) e[i].clear();
			for (auto [u, v, w, c] : edges)
			{
				e[u].push_back({v, w, c, (int)e[v].size()});
				e[v].push_back({u, 0, -c, (int)e[u].size() - 1});
			}
			static ll h[N];
			auto get_h = [&]() {
				fill_n(h, n, inf);
				memset(ed, 0, n * sizeof ed[0]);
				queue<int> q;
				q.push(s); h[s] = 0;
				while (q.size())
				{
					int u = q.front(); q.pop(); ed[u] = 0;
					for (auto [v, w, c, id] : e[u]) if (w && h[v] > h[u] + c)
					{
						assert(c >= 0);
						h[v] = h[u] + c;
						if (!ed[v]) q.push(v), ed[v] = 1;
					}
				}
				return;
			};
			auto dijkstra = [&]() -> pair<ll, lll> {
				static int fl[N], zl[N];
				int i;
				memset(ed, 0, n * sizeof ed[0]);
				fill_n(dis, n, inf);
				using pa = pair<ll, int>;
				priority_queue<pa, vector<pa>, greater<pa>> q;
				dis[s] = 0; q.push({0, s});
				while (q.size())
				{
					int u = q.top().second;
					q.pop(); ed[u] = 1;
					i = 0;
					for (auto [v, w, c, id] : e[u])
					{
						if (w && cmin(dis[v], dis[u] + c))
						{
							assert(c >= 0);
							fl[v] = id, zl[v] = i, pre[v] = u;
							q.push({dis[v], v});
						}
						++i;
					}
					while (q.size() && ed[q.top().second]) q.pop();
				}
				if (dis[t] == inf) return {0, 0};
				ll tf = numeric_limits<ll>::max();
				for (i = t; i != s; i = pre[i]) tf = min(tf, e[pre[i]][zl[i]].w);
				for (i = t; i != s; i = pre[i]) e[pre[i]][zl[i]].w -= tf, e[i][fl[i]].w += tf;
				for (int u = 0; u < n; u++) for (auto &[v, w, c, id] : e[u]) c += dis[u] - dis[v];
				return {tf, (lll)tf * (h[t] += dis[t])};
			};
			get_h();
			for (int u = 0; u < n; u++) for (auto &[v, w, c, id] : e[u]) c += h[u] - h[v];
			ll w = 0, dw;
			lll c = 0, dc;
			do
			{
				tie(dw, dc) = dijkstra();
				w += dw, c += dc;
			} while (dw);
			vector<ll> ans, id(n);
			for (auto [u, v, w, c] : edges)
			{
				++id[u];
				assert(e[v][id[v]].v == u);
				ans.push_back(e[v][id[v]++].w);
			}
			return {w, c, ans};
		}
	}
	using cost_flow::mcmf_spfa, cost_flow::mcmf_spfa_scaling, cost_flow::mcmf_dijk;
	namespace bounded_flow
	{
		vector<ll> valid_flow(int n, const vector<tuple<int, int, ll, ll>> &edges)
		{//返回空 vector 表示无解。最好保证 edges 非空。
			int i, m = edges.size();
			assert(m);
			++n;
			ll tot = 0;
			static ll cd[N];
			memset(cd, 0, n * sizeof cd[0]);
			for (auto [u, v, l, r] : edges) cd[u] += l, cd[v] -= l;
			vector<tuple<int, int, ll>> eg;
			eg.reserve(n + m);
			for (i = 0; i < n; i++) if (cd[i] > 0) eg.push_back({i, n + 1, cd[i]}), tot += cd[i];
			else if (cd[i] < 0) eg.push_back({n, i, -cd[i]});
			for (auto [u, v, l, r] : edges) eg.push_back({u, v, r - l});
			auto [w, ans] = flow::max_flow(n + 1, eg, n, n + 1);
			if (tot != w) return { };
			ans.erase(all(ans) - m);
			for (i = 0; i < m; i++) ans[i] += get<2>(edges[i]);
			return ans;
		}
		pair<ll, vector<ll>> valid_flow_st(int n, vector<tuple<int, int, ll, ll>> edges, int s, int t)
		{//返回值 first -1 表示无解
			ll tot = 0;
			for (auto [u, v, l, r] : edges) tot += (u == s) * r;
			edges.push_back({t, s, 0, tot});
			auto ans = valid_flow(n, edges);
			if (!ans.size()) return {-1, { }};
			ans.pop_back();
			assert(flow::e[s].back().v == t);
			assert(flow::e[t].back().v == s);
			return {flow::e[s].back().w, ans};
		}
		pair<ll, vector<ll>> valid_max_flow(int n, const vector<tuple<int, int, ll, ll>> &edges, int s, int t)
		{//返回值 first -1 表示无解
			auto [r, _] = valid_flow_st(n, edges, s, t);
			if (r == -1) return {-1, { }};
			flow::s = s, flow::t = t;
			flow::e[s].pop_back(), flow::e[t].pop_back();
			while (flow::bfs()) r += flow::dfs(s, flow::inf);
			int m = edges.size(), i;
			vector<ll> ans(m), id(n + 1);
			for (i = 0; i <= n; i++) id[i] = flow::e[i].size();
			for (i = m - 1; i >= 0; i--)
			{
				auto [u, v, l, r] = edges[i];
				--id[u]; ans[i] = flow::e[v][--id[v]].w + l;
			}
			return {r, ans};
		}
		pair<ll, vector<ll>> valid_min_flow(int n, const vector<tuple<int, int, ll, ll>> &edges, int s, int t)
		{
			auto [r, _] = valid_flow_st(n, edges, s, t);
			if (r == -1) return {-1, { }};
			flow::s = t; flow::t = s;
			flow::e[s].pop_back(); flow::e[t].pop_back();
			while (flow::bfs()) r -= flow::dfs(t, flow::inf);
			int m = edges.size(), i;
			vector<ll> ans(m), id(n + 1);
			for (i = 0; i <= n; i++) id[i] = flow::e[i].size();
			for (i = m - 1; i >= 0; i--)
			{
				auto [u, v, l, r] = edges[i];
				--id[u]; ans[i] = flow::e[v][--id[v]].w + l;
			}
			return {r, ans};
		}//not check
	}
	using bounded_flow::valid_flow, bounded_flow::valid_flow_st, bounded_flow::valid_max_flow, bounded_flow::valid_min_flow;
	namespace bounded_cost_flow
	{
		tuple<ll, lll, vector<ll>> valid_mcf(int n, const vector<tuple<int, int, ll, ll, ll>> &edges, int s, int t)
		{//[u,v,l,r,c],mincost flow
			++n;
			int ss = n, tt = n + 1, m = edges.size(), i;
			static ll cd[N];
			memset(cd, 0, n * sizeof cd[0]);
			for (auto [u, v, l, r, c] : edges) cd[u] += l, cd[v] -= l;
			vector<tuple<int, int, ll, ll>> e; e.reserve(n + m + 1);
			ll t1 = 0, t2 = 0;
			for (auto [u, v, l, r, c] : edges) e.push_back({u, v, r - l, c});
			for (i = 0; i < n; i++) if (cd[i] > 0) e.push_back({i, tt, cd[i], 0}), t2 += cd[i];
			else if (cd[i] < 0) e.push_back({ss, i, -cd[i], 0});
			for (auto [u, v, w, c] : e) t1 += (u == s) * w;
			e.push_back({t, s, t1, 0});
			auto [tw, tc, _] = mcmf_spfa_scaling(tt, e, ss, tt);//checked spfa/dijk
			if (tw != t2) return {-1, -1, { }};
			tw = cost_flow::e[s].back().w;
			for (auto [u, v, l, r, c] : edges) tc += (lll)l * c;
			vector<ll> ans(m), id(n);
			for (i = 0; i < m; i++)
			{
				auto [u, v, l, r, c] = edges[i];
				assert(cost_flow::e[v][id[v]].v == u);
				++id[u]; ans[i] = cost_flow::e[v][id[v]++].w + l;
			}
			return {tw, tc, ans};
		}
		tuple<ll, lll, vector<ll>> valid_mcmf(int n, const vector<tuple<int, int, ll, ll, ll>> &edges, int s, int t)
		{//[u,v,l,r,c],mincost max_flow, not checked dijk
			auto [tw, tc, _] = valid_mcf(n, edges, s, t);
			if (tw == -1) return {-1, -1, { }};
			cost_flow::e[s].pop_back();
			cost_flow::e[t].pop_back();
			cost_flow::s = s; cost_flow::t = t;
			ll dw;
			lll dc;
			do
			{
				tie(dw, dc) = cost_flow::spfa();
				tw += dw; tc += dc;
			} while (dw);
			int m = edges.size(), i;
			vector<ll> ans(m), id(n + 1);
			for (i = 0; i < m; i++)
			{
				auto [u, v, l, r, c] = edges[i];
				++id[u]; ans[i] = cost_flow::e[v][id[v]++].w + l;
			}
			return {tw, tc, ans};
		}
		tuple<ll, lll, vector<ll>> valid_mcmf_scaling(int n, const vector<tuple<int, int, ll, ll, ll>> &edges, int s, int t)
		{//[u,v,l,r,c],mincost max_flow, not checked dijk
			using cost_flow::e, cost_flow::spfa_loop;
			auto [tw, tc, _] = valid_mcf(n, edges, s, t);
			if (tw == -1) return {-1, -1, { }};
			e[s].pop_back();
			e[t].pop_back();
			cost_flow::s = s; cost_flow::t = t;
			n = cost_flow::n;
			int m = edges.size(), i, j;
			vector<ll> cur;
			ll tot = 1;
			for (i = 0; i < n; i++) for (auto [v, w, c, id] : cost_flow::e[i]) tot += w;
			for (i = 0; i < n; i++)
			{
				for (auto &[v, w, c, id] : cost_flow::e[i])
					cur.push_back(w), w = 0;
				if (i == t) cur.push_back(tot);
				if (i == s) cur.push_back(0);
			}
			e[t].push_back({s, 0, -cost_flow::inf, (int)e[s].size()});
			e[s].push_back({t, 0, cost_flow::inf, (int)e[t].size() - 1});
			ll w = 0, dw;
			lll c = 0, dc;
			for (int g = __lg(*max_element(all(cur))); g >= 0; g--)
			{
				for (i = j = 0; i < n; i++) for (auto &[v, w, c, id] : e[i]) w = w * 2 + (cur[j++] >> g & 1);
				w *= 2, c *= 2;
				do
				{
					tie(dw, dc) = spfa_loop();
					w += dw, c += dc;
				} while (dw || dc < 0);
			}
			vector<ll> ans(m), id(n + 1);//方案很可能完全不对
			for (i = 0; i < m; i++)
			{
				auto [u, v, l, r, c] = edges[i];
				++id[u]; ans[i] = cost_flow::e[v][id[v]++].w + l;
			}
			return {tw + w, tc + c, ans};
		}
	}
	using bounded_cost_flow::valid_mcf, bounded_cost_flow::valid_mcmf, bounded_cost_flow::valid_mcmf_scaling;
	namespace ne_cost_flow
	{
		tuple<ll, lll, vector<ll>> ne_mcmf(int n, const vector<tuple<int, int, ll, ll>> &edges, int s, int t)
		{
			vector<tuple<int, int, ll, ll, ll>> e;
			for (auto [u, v, w, c] : edges) if (c >= 0) e.push_back({u, v, 0, w, c}); else
			{
				e.push_back({u, v, w, w, c});
				e.push_back({v, u, 0, w, -c});
			}
			auto [tw, tc, res] = valid_mcmf_scaling(n, e, s, t);
			int m = edges.size(), i, j;
			vector<ll> ans(m);
			for (i = j = 0; i < m; i++, j++)
			{
				auto [u, v, w, c] = edges[i];
				if (c >= 0) ans[i] = res[j];
				else ans[i] = w - res[++j];
			}
			assert(j == e.size());
			return {tw, tc, ans};
		}
		tuple<ll, lll, vector<ll>> ne_valid_mcf(int n, const vector<tuple<int, int, ll, ll, ll>> &edges, int s, int t)
		{
			vector<tuple<int, int, ll, ll, ll>> e;
			for (auto [u, v, l, r, c] : edges) if (c >= 0) e.push_back({u, v, l, r, c}); else
			{
				e.push_back({u, v, r, r, c});
				e.push_back({v, u, 0, r - l, -c});
			}
			auto [tw, tc, res] = valid_mcf(n, e, s, t);
			if (tw == -1) return {-1, -1, { }};
			int m = edges.size(), i, j;
			vector<ll> ans(m);
			for (i = j = 0; i < m; i++, j++)
			{
				auto [u, v, l, r, c] = edges[i];
				if (c >= 0) ans[i] = res[j];
				else ans[i] = r - res[++j];
			}
			assert(j == e.size());
			return {tw, tc, ans};
		}
	}
	using ne_cost_flow::ne_mcmf, ne_cost_flow::ne_valid_mcf;
}
