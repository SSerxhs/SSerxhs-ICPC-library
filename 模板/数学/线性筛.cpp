namespace Prime
{
	typedef unsigned int ui;
	typedef unsigned long long ll;
	const int N=1e6+2;
	const ll M=(ll)(N-1)*(N-1);
	ui pr[N],mn[N],phi[N],cnt;
	int mu[N];
	void init_prime()
	{
		ui i,j,k;
		phi[1]=mu[1]=1;
		for (i=2;i<N;i++)
		{
			if (!mn[i])
			{
				pr[cnt++]=i;
				phi[i]=i-1;mu[i]=-1;
				mn[i]=i;
			}
			for (j=0;(k=i*pr[j])<N;j++)
			{
				mn[k]=pr[j];
				if (i%pr[j]==0)
				{
					phi[k]=phi[i]*pr[j];
					break;
				}
				phi[k]=phi[i]*(pr[j]-1);
				mu[k]=-mu[i];
			}
		}
		//for (i=2;i<N;i++) if (mu[i]<0) mu[i]+=p;
	}
	template<typename T> T getphi(T x)
	{
		assert(M>=x);
		T r=x;
		for (ui i=0;i<cnt&&(T)pr[i]*pr[i]<=x&&x>=N;i++) if (x%pr[i]==0)
		{
			ui y=pr[i],tmp;
			x/=y;
			while (x==(tmp=x/y)*y) x=tmp;
			r=r/y*(y-1);
		}
		if (x>=N) return r/x*(x-1);
		while (x>1)
		{
			ui y=mn[x],tmp;
			x/=y;
			while (x==(tmp=x/y)*y) x=tmp;
			r=r/y*(y-1);
		}
		return r;
	}
	template<typename T> vector<pair<T,ui>> getw(T x)
	{
		assert(M>=x);
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
using Prime::pr,Prime::phi,Prime::getw;
using Prime::mu,Prime::init_prime;