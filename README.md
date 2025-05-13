# ece228-homework-2-solved
**TO GET THIS SOLUTION VISIT:** [ECE228 Homework 2 Solved](https://www.ankitcodinghub.com/product/ece228-homework-2-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;92377&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;ECE228 Homework 2 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
&nbsp;

<strong>PART II: Programming Assignment</strong>

<h1>1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Problem 1: Policy Gradient</h1>
Experiment with policy gradient and its variants for both continuous and discrete environments. The starter code is setup in main.py, and everything that you need to implement is in the files network utils.py, policy.py, policy gradient.py and baseline network.py. The file has detailed instructions for each implementation task, but an overview of key steps in the algorithm is provided here.

<h2>Review of the REINFORCE Algorithm</h2>
<strong>Policy Gradient: </strong>recall the policy gradient theorem,

where <em>τ </em>denotes the trajectory, <em>G</em>(<em>τ</em>) is the return of trajectory <em>τ</em>. The second line uses |<em>D</em>| sampled trajectories for estimating the expectation, where <em>r</em>(<em>s<sup>i</sup><sub>t</sub>,a<sup>i</sup><sub>t</sub></em>) is the immediate return for taking action <em>a<sup>i</sup><sub>t </sub></em>at state <em>s<sup>i</sup><sub>t</sub></em>, and ) being the sampled return of trajectory <em>i</em>.

<strong>REINFORCE: </strong>Recall that the policy at time <em>t</em><sup>′ </sup>does not affect the reward at time <em>t </em>when <em>t &lt; t</em><sup>′</sup>. Thus, the REINFORCE estimator can be expressed as the gradient of the following objective function:

where <em>D </em>is the set of all trajectories. If different trajectories have different length <em>H<sub>i </sub></em>(with <em>T </em>being the maximum trajectory length, i.e., <em>H<sub>i </sub></em>≤ <em>T,</em>∀<em>i</em>), we can re-write (1) as follows,

<em>,&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </em>(2)

return:<em>G<sup>i</sup><sub>t</sub></em>

where <em>D </em>is the set of all trajectories collected by policy

is trajectory <em>i </em>and <em>H<sub>i </sub></em>is the length of trajectory <em>i</em>. For your implementation of the REINFORCE algorithm, please follow equation (2).

<h2>Instructions on the Implementation</h2>
The functions that you need to implement in network utils.py, policy.py, policy gradient.py, baseline network.py are enumerated here. Detailed instructions for each function can be found in the comments in each of these files.

Note: The “batch size” for all the arguments is <sup>P</sup><em>H<sub>i </sub></em>since we already flattened out all the episode observations, actions and rewards for you.

In network utils.py,

<ul>
<li>build mlp</li>
</ul>
In policy.py,

<ul>
<li>act</li>
<li>action distribution</li>
<li>init</li>
<li>std</li>
<li>action distribution</li>
</ul>
In policy gradient.py,

<ul>
<li>init policy</li>
<li>get returns</li>
<li>normalize advantage</li>
<li>update policy</li>
</ul>
In baseline network.py,

<ul>
<li>init</li>
<li>forward</li>
<li>calculate advantage</li>
<li>update baseline</li>
</ul>
<ol>
<li><strong>: Compute Return </strong>To compute the REINFORCE estimator, you will need to calculate the values (we drop the trajectory index <em>i </em>for simplicity), where</li>
</ol>
<em>H</em>

<em>G</em><em>t </em>= X<em>γ</em><em>t</em>′−<em>t</em><em>r</em><em>t</em>′

<em>t</em><sup>′</sup>=<em>t</em>

Naively, computing all these values takes <em>O</em>(<em>H</em><sup>2</sup>) time. Describe how to compute them in <em>O</em>(<em>H</em>) time.

<strong>: REINFORCE Algorithm Implementation </strong>Implement the REINFORCE algorithm and test it on three OpenAI Gym environments: cartpole, pendulum and cheetah. We have provided some basic tests to sanity check your implementation.

Please note that the tests are not comprehensive, and passing them does not guarantee a correct implementation. Use the following command to run the tests:

python run basic tests.py

You can also add additional tests of your own design in tests/test basic.py. We have the following expectation about performance to receive full credit:

<ul>
<li>cartpole: Should reach the max reward of 200 (although it may not stay there)</li>
<li>pendulum: Should reach the max reward of 1000 (although it may not stay there)</li>
<li>cheetah: Should reach at least 200 (Could be as large as 950)</li>
</ul>
<strong>Note: </strong>In order to get good performance, you may need to adjust hyperparameters such as initialization of logstd, architecture configuration, learning rate, etc.

<strong>iii. : Baseline Subtraction </strong>One difficulty of training with the REINFORCE algorithm is the sampled return(s) <em>G<sup>i</sup><sub>t </sub></em>can have high variance. To reduce variance, we subtract a baseline <em>b<sub>ϕ</sub></em>(<em>s</em>) from the estimated returns when computing the policy gradient. A good baseline is the state value function, <em>V <sup>π</sup></em><em><sup>θ</sup></em>(<em>s</em>), which requires a training update to <em>ϕ </em>to minimize the following mean-squared error loss:

The general form for running your REINFORCE implementation is as follows:

python main.py –env-name ENV –seed SEES –no-baseline

if not using a baseline, or

python main.py –env-name ENV –seed SEES –baseline

if using a baseline. Here ENV should be cartpole, pendulum or, cheetah, and SEED should be a positive integer.

For each of the 3 environments, choose 3 random seeds and run the algorithm with both without baseline and with baseline. Then plot the result using

python plot.py –env-name ENV –seeds SEEDS

where SEEDS should be comma-separated list of seeds which you want to plot (e.g. –seeds 1,2,3).

Please include the plots (one for each environment) in your notebook, and comment on whether or not you observe improved performance using a baseline.

<strong>2 Practice Problem, You don’t need to submit this: Value iteration and Policy Iteration.</strong>

In this problem you will program value iteration and policy iteration. You will use environments implementing the OpenAI Gym Environment API. For more information on Gym and the API, see https://gym.openai.com/. We will be working with different versions of Frozen Lake environments.

In this domain, the agent starts at a fixed starting position, marked with “S”. The agent can move up, down, left and right. In deterministic versions, the up action action will always move the agent up, left will always move left, etc. In the stochastic versions, the up action will move up with a probability of 1/3, left with a probability of 1/3 and right with a probability of 1/3.

There are three different tile types: frozen, hole and ground. When the agent lands on a frozen tile, it receives 0 reward. When the agent lands on a hole tile it receives 0 reward and the episode ends. Then the agent lands on the goal tile, it receives +1 reward and the episode ends. We have provided you 2 different maps. An invalid action (e.g. towards the boundary) will keep the agent where it is.

States are represented as integers numbered from left to right, top to bottom starting from 0. So the upper left corner of the 4×4 map is state 0, the bottom right corner is state

15.

You will implement value iteration and policy iteration using the provided environments. Some function templates are provided for you to fill in. Specific coding instructions are provided in the source code files.

Be careful implementing value iteration and policy evaluation. Keep in mind that in this environment the reward function depends on the current state, the current action and the next state. Also terminal states are slightly different.

<ol>
<li><strong>&nbsp;Deterministic Frozen Lake </strong>Answer the following questions for the maps Deterministic-4×4-FrozenLake-v0 and Deterministic-8×8-FrozenLake-v0.
<ol>
<li>Using the environment, find the optimal policy iteration. Record the time taken for execution, then number of policy improvement steps and the total number of policy evaluation steps. Use <em>γ </em>= 0<em>.</em> Use a stopping tolerance of 10<sup>−3 </sup>for the policy evaluations.</li>
<li>What is the optimal policy for this map? Show as a grid of letters with “U”, “D”, “L”, “R” representing the actions up, down, left, right respectively. An example output is shown below.</li>
<li>Find the value function of this policy. Plot it as a color image as shown, where each square has its value as a color.</li>
<li>Find the optimal value function directly using the value iteration. Record the time taken for execution, and the number of iterations required. Use <em>γ </em>= 0<em>.</em>9, a stopping tolerance of 10<sup>−3</sup>.</li>
<li>Plot this value function as a color image, where each square shows its value as a color.</li>
<li>Which algorithm was faster? Which took less iterations?</li>
<li>Are there any differences in the value function?</li>
<li>Convert the optimal value function to the optimal policy. Show as a grid of letters with “U”, “D”, “L”, “R” representing the actions up, down, left, right respectively.</li>
<li>Write an agent that executes the optimal policy. Record the total cumulative discounted reward. Does the value match the value computed for the starting state? If not, explain why.</li>
</ol>
</li>
</ol>
Figure 1: Example policy for FrozenLake-v0

<ol>
<li><strong>[0 points]: Stochastic Frozen Lake </strong>Answer the following questions for the map Stochastic-8×8-FrozenLake-v0.</li>
<li>Using value iteration, find the optimal value function. Record the time taken for execution, then number of policy improvement steps and the total number of policy evaluation steps. Use <em>γ </em>= 0<em>.</em> Use a stopping tolerance of 10<sup>−3</sup>.</li>
<li>Plot the value function as a color map. Is the value function different compared to deterministic versions of the maps?</li>
<li>Convert this value function to the optimal policy and include it in the notebook.</li>
<li>Does the optimal policy differ from the optimal policy in the deterministic maps? If so, pick a state where the policy differs and explain why the action is different.</li>
<li>Write an agent that executes the optimal policy. Run this agent 100 times on the map and record the total cumulative discounted reward. Average the results. Does this value match the value computed for the starting state? If not, explain why.</li>
</ol>
Figure 2: Example of value function color plot. Make sure you include the color bar or some kind of key.
