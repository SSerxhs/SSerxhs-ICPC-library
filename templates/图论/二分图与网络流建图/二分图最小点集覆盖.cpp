#include "bits/stdc++.h"
using namespace std;
const int N = 5e3 + 2;
vector<int> e[N];
int ed[N], lk[N], kl[N], flg[N], now;
bool dfs(int u)
{
	for (int v : e[u]) if (ed[v] != now)
	{
		ed[v] = now;
		if (!lk[v] || dfs(lk[v])) return lk[v] = u;
	}
	return 0;
}
void dfs2(int u)
{
	for (int v : e[u]) if (!flg[v]) flg[v] = 1, dfs2(lk[v]);
}
int main()
{
	int n, m, i, r = 0;
	cin >> n >> m;
	while (m--)
	{
		int u, v;
		cin >> u >> v;
		e[u].push_back(v);
	}
	for (i = 1; i <= n; i++) dfs(now = i);
	for (i = 1; i <= n; i++) kl[lk[i]] = i;
	for (i = 1; i <= n; i++) if (!kl[i]) dfs2(i);
	vector<int> A[2];
	for (i = 1; i <= n; i++) if (lk[i])
	{
		if (flg[i]) A[1].push_back(i); else A[0].push_back(lk[i]);
	}
	for (int j = 0; j < 2; j++)
	{
		cout << A[j].size();
		for (int x : A[j]) cout << ' ' << x; cout << '\n';
	}
}
