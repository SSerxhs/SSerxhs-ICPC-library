//http://10.49.18.71/submission/164696
#include<bits/stdc++.h>
using namespace std;
constexpr int M = 1e6 + 5;
int n, q, a[M], tr[M], ans[M];
vector<pair<int, int>>qry[M];
int blk, bel[M], L[M], R[M], val[M];
priority_queue<int>Q[M];
priority_queue<int, vector<int>, greater<int>>P[M];
int read() {
	int x = 0; char ch = getchar();
	while (!isdigit(ch)) ch = getchar();
	while (isdigit(ch)) x = x * 10 + ch - 48, ch = getchar();
	return x;
}
void update(int x) { while (x) tr[x]++, x -= x & -x; }
int query(int x) { int res = 0; while (x <= n) res += tr[x], x += x & -x; return res; }
int main() {
	n = read(); q = read();
	for (int i = 1; i <= n; i++) a[i] = read() + 1;
	blk = (int)ceil(sqrt(n));
	for (int i = 1; i <= n; i++) bel[i] = (i - 1) / blk + 1;
	for (int i = 1; i <= bel[n]; i++) L[i] = R[i - 1] + 1, R[i] = R[i - 1] + blk; R[bel[n]] = n;
	auto push_back = [&](int x) {
		const int p = a[x], B = bel[p];
		if (!P[B].empty()) {
			for (int i = L[B]; i <= R[B]; i++)
				if (val[i]) {
					P[B].push(val[i]);
					val[i] = P[B].top();
					P[B].pop();
				}
			while (!P[B].empty()) P[B].pop();
		}
		val[p] = x; Q[B].push(x);
		int tmp = 0; bool flag = 0;
		for (int i = p + 1; i <= R[B]; i++)
			if (tmp < val[i]) swap(tmp, val[i]), flag = 1;
		if (flag) {
			while (!Q[B].empty()) Q[B].pop();
			for (int i = L[B]; i <= R[B]; i++)
				if (val[i]) Q[B].push(val[i]);
		}
		for (int i = B + 1; i <= bel[n]; i++)
			if (!Q[i].empty() && tmp < Q[i].top()) {
				P[i].push(tmp);
				if (tmp) Q[i].push(tmp);
				tmp = Q[i].top(), Q[i].pop();
			}
		update(tmp);
	};
	for (int i = 1; i <= q; i++) {
		int l = read() + 1, r = read();
		ans[i] = r - l + 1;
		if (l <= r)qry[r].emplace_back(l, i);
	}
	for (int i = 1; i <= n; i++) {
		push_back(i);
		for (auto [x, id] : qry[i])
			ans[id] -= query(x);
	}
	for (int i = 1; i <= q; i++) printf("%d\n", ans[i]);
	return 0;
}
