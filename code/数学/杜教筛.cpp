namespace du_seive
{
	using ui = unsigned int;
	using ull = unsigned long long;
	unordered_map<ull, ui> mp;
	const int N = 1e7 + 2;
	const ui p = 998244353;
	ui pr[N], phi[N];
	ui cnt;
	void init()
	{
		cnt = 0; phi[1] = 1;
		int i, j;
		for (i = 2; i < N; i++)
		{
			if (!phi[i])
			{
				pr[cnt++] = i;
				phi[i] = i - 1;
			}
			for (j = 0; i * pr[j] < N; j++)
			{
				if (i % pr[j] == 0)
				{
					phi[i * pr[j]] = phi[i] * pr[j];
					break;
				}
				phi[i * pr[j]] = phi[i] * (pr[j] - 1);
			}
			if ((phi[i] += phi[i - 1]) >= p) phi[i] -= p;
		}
	}
	ui get_phi_sum(ull n)
	{
		if (n < N) return phi[n];
		if (mp.count(n)) return mp[n];
		ui sum = 0;
		for (ull i = 2, j, k; i <= n; i = j + 1)
		{
			j = n / (k = n / i);
			sum = (sum + (ull)get_phi_sum(k) * (j - i + 1)) % p;
		}
		ui nn = n % p;
		sum = (nn * (nn + 1llu) / 2 + p - sum) % p;
		mp[n] = sum;
		return sum;
	}
	int _ = (init(), 0);
}
using du_seive::get_phi_sum;
