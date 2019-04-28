 import java.util.*;

 public class heredity
 {
 	private int genelenth;//基因长度
 	private int popsize;//种群数量
 	private int maxgens;//迭代次数
 	private double pxover;//交叉概率
 	private double pmultation;//变异概率
 	private double sumfitness;//总适应度
 	private	int ncross=0;
 	
 	private class chromosome//个体属性
 	{
 		int gene []=new int [genelenth];//基因序列
 		double fitness;//该染色的基因适应度
 		boolean isselect;//该染色体是否被选择进入下一代
 		double execeptp;//该染色体期望
 		double selectp;//该染色体的选择概率
 	}
 	private chromosome child[];//子代
  	private chromosome chom[];//父代
 	public heredity(int gl,int pop,int mg,double px,double pm)//构造函数初始化基因长度，种群数量，迭代次数，交叉概率，变异概率
 	{
 		genelenth=gl;
 		popsize=pop;
 		maxgens=mg;
 		pxover=px;
 		pmultation=pm;
        child=new chromosome [popsize];
 		chom=new chromosome [popsize];
 		for (int i=0;i<popsize;i++)
 		{
 			child[i]=new chromosome();
 			chom[i] = new chromosome();
 			chom[i].fitness=0;
 			chom[i].execeptp=0;
 			chom[i].isselect=true;
 			chom[i].selectp=0;
 			for  (int j=0;j<genelenth;j++)//基因序列赋值
 			{
 				 chom[i].gene[j]=flip();
 			}
 		}
 	}
 	public int flip()
 	{
 		float a=0.5f;
 		if (Math.random()>a)
 		  return (1);
 		else
 		  return (0);
	}
 	public void calall()// 计算染色体的适应度，选择概率，期望概率，是否被选择进入下一代的概率
 	{
 	  for (int i=0;i<popsize;i++)
 		{
 			chom[i].fitness=0;
 			chom[i].execeptp=0;
 			chom[i].isselect=true;
 			chom[i].selectp=0;
 		}
 		calfitness(chom);
 		calselectp();
 		isenter();
 	}
 	public void crossover(int x,int y)//交叉
 	{
 		int jcross=0;
		if (Math.random()<=pxover)
		 	{
		 	    jcross=(int)((Math.random()*(genelenth-2))+1);//产生[1,21)交叉点
		 		
		 		for(int i=1;i<=genelenth;i++)
		 			{
		 			 	if (jcross>=i)
		 				   {
		 					 	child[ncross].gene[i-1]=chom[x].gene[i-1];
		 						child[ncross+1].gene[i-1]=chom[y].gene[i-1];
		 					}
		 				else
		 			  		if ((jcross>1)&&(jcross<=genelenth))
		 			 		{
		 						child[ncross].gene[i-1]=chom[y].gene[i-1];
		 						child[ncross+1].gene[i-1]=chom[x].gene[i-1];
		 			  		}
		 			  
		 			}		
		 	}
		else
			 for (int i=0;i<genelenth;i++)
				{
				   child[ncross].gene[i]=chom[x].gene[i];
				   child[ncross+1].gene[i]=chom[y].gene[i];
				 }
		variation(ncross);
 		variation(ncross+1);
		ncross+=2;
	}
 	public int select()//选择
 	{
 		int i=0;
 		double sum=0,pick=0;
 		pick=Math.random();
 		for (;(sum<pick)&&(i<popsize);i++)
 		{
           sum+=chom[i].selectp;
 		}
 		return (i-1);
 	}
public void variation(int i)//变异
 	{
 		 double random;
 		 int var1;
 		 	random=Math.random();// 产生[0,1) 随机数
 		 	if (random<=pmultation)
 		 	{
 		 		var1=(int)(Math.random()*genelenth);
 		 		if(child[i].gene[var1]==1)
 		 		   child[i].gene[var1]=0;
 		 		 else
 		 		   child[i].gene[var1]=1;
 		 	}
 		 	isenter();
 	}
 	public void isenter()//是否进入下一代
 	{
 		for (int i=0;i<popsize;i++)
 		{
 			for (int j=0;j<popsize;j++)
 			{
 				if ((chom[i].fitness<child[j].fitness)&&child[j].isselect)
 				{
 					chom[i]=child[j];
 					child[j].isselect=false;
 					break;
 				}
 				else
 				   continue;
 			}
 		}
 	}
 	public void generation()//选择，交叉，变异
 	{
 		int mate1=0,mate2=0,j=0;
 		do{
 			mate1=select();
 			mate2=select();
 			crossover(mate1,mate2);
 			calfitness(child);
 			j=j+1;
 		}while(j<(popsize/2));
 		isenter();
 		for (int i=0;i<popsize;i++)
 		   child[i].isselect=true;
 		ncross=0;
 	}
 	public void calfitness(chromosome chom1[])// 计算适应度 
 	{
 		double k=0.0;;
 		double temp=0.0;
 		for (int i=0;i<popsize;i++)
 		{
 			for (int j=0;j<genelenth;j++)
 			{
 		        k+=chom1[i].gene[j]*Math.pow(2.0,(double)j);
 		    }
 		   temp=k*(3.0/(Math.pow(2.0,(double)genelenth)-1))-1.0;
 		   chom1[i].fitness=temp*Math.sin(10*Math.PI*temp)+2.0;
 		   k=0;
 		}
 	}
 	public void calselectp()//计算选择概率
 	{
 		double sum=0;
 		for (int i=0;i<popsize;i++)
 			sum+=chom[i].fitness;
 		sumfitness = sum;//获取总适应度
 		for (int j=0;j<popsize;j++)
 		    chom[j].selectp=(double)(chom[j].fitness/sum);
 	}
    public static void main(String [] args)//主函数
 	{
 		int dai=0;
  		heredity test=new heredity(22,200,500,0.25,0.03);
  		while (test.maxgens>0)
  		{
  			test.calall();
  			test.generation();
  			test.maxgens--;
  			dai=500-test.maxgens;
  			double temp=test.chom[0].fitness;
            for (int i=0;i<test.popsize;i++)
            	if (test.chom[i].fitness<temp)
            	   temp=test.chom[i].fitness;
			System.out.println("----------第"+dai+"代-----------   "+temp);

  		}
  	
 	}
 }