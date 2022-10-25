namespace Prime
{
	typedef unsigned int ui;
	typedef unsigned long long ll;
	const int N=1e7+2;
	ui pr[N],mn[N],cnt;
	void init_prime()
	{
		ui i,j,k;
		for (i=2;i<N;i++)
		{
			if (!mn[i])
			{
				pr[cnt++]=i;
				mn[i]=i;
			}
			for (j=0;(k=i*pr[j])<N;j++)
			{
				mn[k]=pr[j];
				if (i%pr[j]==0) break;
			}
		}
		//for (i=2;i<N;i++) if (mu[i]<0) mu[i]+=p;
	}
	template<typename T> vector<pair<T,ui>> getw(T x)
	{
		assert((ll)(N-1)*(N-1)>=x);
		vector<pair<T,ui>> r;
		for (ui i=0;i<cnt&&(T)pr[i]*pr[i]<=x&&x>=N;i++) if (x%pr[i]==0)
		{
			ui y=pr[i],z=1,tmp;
			x/=y;
			while (x==(tmp=x/y)*y) x=tmp,++z;
			r.push_back({y,z});
		}
		if (x>=N)
		{
			r.push_back({x,1});
			return r;
		}
		while (x>1)
		{
			ui y=mn[x],z=1,tmp;
			x/=y;
			while (x==(tmp=x/y)*y) x=tmp,++z;
			r.push_back({y,z});
		}
		return r;
	}
}
using Prime::pr,Prime::getw;
using Prime::init_prime;