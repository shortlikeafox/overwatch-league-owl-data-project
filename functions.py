import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Helpful functions for the project


#Input: Master DF
#Output: Master DF with winner_label column filled in to show who won team one or team twol

def set_winners(master_df):
    id_list = master_df.id.unique()

    for i in id_list:
        t1 = ((master_df[master_df['id'] == i]).team_one).iloc[0]
        t2 = ((master_df[master_df['id'] == i]).team_two).iloc[0]
        winner = ((master_df[master_df['id'] == i]).winner).iloc[0]
        mask = master_df['id'] == i
        if t1 == winner:
            master_df['winner_label'][mask] = 0
        elif t2 == winner:
            master_df['winner_label'][mask] = 1
    
    return master_df







#Very specific helper function
#returns a list of the following information: 
#Returned {team_one_wins, team_one_losses, team_one_total_matches, team_two_wins, team_two_losses, team_two_total_matches}
#Takes the master_df and a match ID as the input

def get_win_loss_total(wl_df, wl_id):
    t1 = ((wl_df[wl_df['id'] == wl_id]).team_one).iloc[0]
    t2 = ((wl_df[wl_df['id'] == wl_id]).team_two).iloc[0]
    date = ((wl_df[wl_df['id'] == wl_id]).date).iloc[0]
    #print(f"Team 1: {t1} Team 2: {t2} Date: {date}")
    teams=[t1, t2]
    
    return_info=[]
    
    for team in teams:
        df_partial_season=wl_df[wl_df['date']<date]
        season_1 = df_partial_season[df_partial_season['team_one'] == team]
        season_2 = df_partial_season[df_partial_season['team_two'] == team]
        team_season = season_1.append(season_2)
        wins = len(team_season[team_season['winner'] == team])
        losses = len(team_season[team_season['winner'] != team])
        matches = wins+losses
        #print(f"{team} {len(season_1)} {len(season_2)} {len(team_season)} {wins} ")
        return_info.append(wins)
        return_info.append(losses)
        return_info.append(matches)
        
    return(return_info)






#This data isn't in the Blizzard information dump so it has to be hardcoded
#Input: master_df
#output: master_df with last season's results
def find_last_season_results(master_df):
    results_2018 = {'Philadelphia Fusion':6, 
                   'New York Excelsior':1, 
                   'Seoul Dynasty':8,
                   'Shanghai Dragons':12,
                   'Toronto Defiant': '', 
                   'Atlanta Reign': '', 
                   'Dallas Fuel': 10,
                   'Chengdu Hunters': '', 
                   'London Spitfire': 5, 
                   'Washington Justice': '',
                   'Los Angeles Valiant': 2, 
                   'Vancouver Titans': '', 
                   'Houston Outlaws': 7,
                   'San Francisco Shock': 9, 
                   'Guangzhou Charge': '', 
                   'Los Angeles Gladiators': 4,
                   'Hangzhou Spark': '', 
                   'Florida Mayhem': 11, 
                   'Paris Eternal': '', 
                   'Boston Uprising': 3     
    }
    results_2019 = {
        'Toronto Defiant':17, 'London Spitfire':7, 'Los Angeles Gladiators':5,
 'Los Angeles Valiant':13, 'Boston Uprising':19, 'San Francisco Shock':3,
 'Florida Mayhem':20, 'Washington Justice':17, 'New York Excelsior':2,
 'Paris Eternal':14, 'Guangzhou Charge':9, 'Chengdu Hunters':12, 'Hangzhou Spark':4,
 'Seoul Dynasty':8, 'Shanghai Dragons':11, 'Houston Outlaws':16,
 'Philadelphia Fusion':10, 'Dallas Fuel':15, 'Vancouver Titans':1, 'Atlanta Reign':6
        
    }
    
    
    years = ['2019', '2020']
    for year in years:
        if year == '2019':
            min_date = '01/01/2019'
            max_date = '12/31/2019'
            
        elif year == '2020':
            min_date = '01/01/2020'
            max_date = '12/31/2020'
            
        df_reduced = master_df[master_df['date'] > min_date]
        df_reduced = df_reduced[df_reduced['date'] < max_date]        
        #print(df_reduced.team_one.unique())
        id_list=df_reduced.id.unique()
        for i in id_list:
            t1 = ((df_reduced[df_reduced['id'] == i]).team_one).iloc[0]
            t2 = ((df_reduced[df_reduced['id'] == i]).team_two).iloc[0]
            mask = master_df['id'] == i
            if year=='2019':
                master_df['t1_place_last_season'][mask] = results_2018.get(t1)
                master_df['t2_place_last_season'][mask] = results_2018.get(t2)
            if year=='2020':
                master_df['t1_place_last_season'][mask] = results_2019.get(t1)
                master_df['t2_place_last_season'][mask] = results_2019.get(t2)
                
    return master_df



#Input: Master_df
#Output: master_df with head-to-head information filled in

def find_head_to_head_results(master_df):
    id_list = master_df.id.unique()

    for i in id_list:
        t1 = ((master_df[master_df['id'] == i]).team_one).iloc[0]
        t2 = ((master_df[master_df['id'] == i]).team_two).iloc[0]
        date = ((master_df[master_df['id'] == i]).date).iloc[0]
        
        
        #print(f"Team 1: {t1} Team 2: {t2} Date: {date}")

        df_partial_season=master_df[master_df['date']<date]
        season_1 = df_partial_season[df_partial_season['team_one'] == t1]
        season_1 = season_1[season_1['team_two'] == t2]        
        season_2 = df_partial_season[df_partial_season['team_two'] == t1]
        season_2 = season_2[season_2['team_one'] == t2]
        team_season = season_1.append(season_2)
        wins = len(team_season[team_season['winner'] == t1])
        losses = len(team_season[team_season['winner'] != t1])
        matches = wins+losses

        mask = master_df['id'] == i
        master_df['t1_wins_vs_t2'][mask] = wins
        master_df['t1_losses_vs_t2'][mask] = losses
        master_df['t1_matches_vs_t2'][mask] = matches   
        #print(f"{t1} vs {t2} on {date} record: {wins} - {losses} ")
        if matches > 0:
            master_df['t1_win_percent_vs_t2'][mask] = wins/matches

        
    return master_df



#input: #master_df, min_date, max_date, type_option(current_season OR all_time)
def find_team_match_results(master_df, min_date, max_date, type_option):
    df_reduced = master_df[master_df['date'] > min_date]
    df_reduced = df_reduced[df_reduced['date'] < max_date]
    
    #We can use this to make an ID list...
    id_list = df_reduced.id.unique()
    
    for i in id_list:
        win_loss_list = (get_win_loss_total(df_reduced, i))
        mask = master_df['id'] == i

        #calculate the win percentages for each match
        if win_loss_list[2] == 0:
            t1_win_percent = 0
        else:
            t1_win_percent = win_loss_list[0] / win_loss_list[2]
        if win_loss_list[5] == 0:
            t2_win_percent = 0
        else:
            t2_win_percent = win_loss_list[3] / win_loss_list[5]
        
        
        if type_option=='current_season':
            master_df['t1_wins_season'][mask] = win_loss_list[0]
            master_df['t1_losses_season'][mask] = win_loss_list[1]
            master_df['t1_matches_season'][mask] = win_loss_list[2]
            master_df['t1_win_percent_season'][mask] = t1_win_percent
            master_df['t2_wins_season'][mask] = win_loss_list[3]
            master_df['t2_losses_season'][mask] = win_loss_list[4]
            master_df['t2_matches_season'][mask] = win_loss_list[5]  
            master_df['t2_win_percent_season'][mask] = t2_win_percent            
        if type_option=='all_time':
            master_df['t1_wins_alltime'][mask] = win_loss_list[0]
            master_df['t1_losses_alltime'][mask] = win_loss_list[1]
            master_df['t1_matches_alltime'][mask] = win_loss_list[2]
            master_df['t1_win_percent_alltime'][mask] = t1_win_percent
            master_df['t2_wins_alltime'][mask] = win_loss_list[3]
            master_df['t2_losses_alltime'][mask] = win_loss_list[4]
            master_df['t2_matches_alltime'][mask] = win_loss_list[5]  
            master_df['t2_win_percent_alltime'][mask] = t2_win_percent
            
    
                
    return(master_df)



#Input: master_df, number of matches we want to look at
#Output: master_df updated with results filled in

def find_last_n_results(master_df, n):
    id_list = master_df.id.unique()
    for i in id_list:
        t1 = ((master_df[master_df['id'] == i]).team_one).iloc[0]
        t2 = ((master_df[master_df['id'] == i]).team_two).iloc[0]
        date = ((master_df[master_df['id'] == i]).date).iloc[0]

        teams=[t1, t2]
    
        win_loss_list=[]

        for team in teams:

            df_partial_season=master_df[master_df['date']<date]
            season_1 = df_partial_season[df_partial_season['team_one'] == team]
            season_2 = df_partial_season[df_partial_season['team_two'] == team]
                        
            #We need to combine season 1 and season 2 and sort by date
            team_df = pd.concat([season_1, season_2])
            #print(f"season_1: {season_1.shape} season_2: {season_2.shape} team_df: {team_df.shape}")
            team_df.sort_values(by=['date'], inplace=True, ascending=False)
            #display(team_df.head)
            top_n_team_df = team_df.head(n)
            
            wins = len(top_n_team_df[top_n_team_df['winner'] == team])
            losses = len(top_n_team_df[top_n_team_df['winner'] != team])
            matches = wins+losses
            #print(f"{team} {len(season_1)} {len(season_2)} {len(team_season)} {wins} ")
            if matches >= n:
                win_loss_list.append(wins)
                win_loss_list.append(losses)
                win_loss_list.append(matches)
            else:
                win_loss_list.append("")
                win_loss_list.append("")
                win_loss_list.append("")
            
            
        
        #print(return_info)
    
        mask = master_df['id'] == i

        #calculate the win percentages for each match
        if win_loss_list[2] == 0:
            t1_win_percent = 0
        elif win_loss_list[2] == "":
            t1_win_percent = ""
        else:
            t1_win_percent = win_loss_list[0] / win_loss_list[2]
        if win_loss_list[5] == 0:
            t2_win_percent = 0
        elif win_loss_list[5] == "":
            t2_win_percent = ""
        else:
            t2_win_percent = win_loss_list[3] / win_loss_list[5]
        
        if n == 3:
            master_df['t1_wins_last_3'][mask] = win_loss_list[0]
            master_df['t1_losses_last_3'][mask] = win_loss_list[1]
            master_df['t1_win_percent_last_3'][mask] = t1_win_percent
            master_df['t2_wins_last_3'][mask] = win_loss_list[3]
            master_df['t2_losses_last_3'][mask] = win_loss_list[4]
            master_df['t2_win_percent_last_3'][mask] = t2_win_percent
        if n == 5:
            master_df['t1_wins_last_5'][mask] = win_loss_list[0]
            master_df['t1_losses_last_5'][mask] = win_loss_list[1]
            master_df['t1_win_percent_last_5'][mask] = t1_win_percent
            master_df['t2_wins_last_5'][mask] = win_loss_list[3]
            master_df['t2_losses_last_5'][mask] = win_loss_list[4]
            master_df['t2_win_percent_last_5'][mask] = t2_win_percent
        elif n == 10:
            master_df['t1_wins_last_10'][mask] = win_loss_list[0]
            master_df['t1_losses_last_10'][mask] = win_loss_list[1]
            master_df['t1_win_percent_last_10'][mask] = t1_win_percent
            master_df['t2_wins_last_10'][mask] = win_loss_list[3]
            master_df['t2_losses_last_10'][mask] = win_loss_list[4]
            master_df['t2_win_percent_last_10'][mask] = t2_win_percent
        elif n == 20:
            master_df['t1_wins_last_20'][mask] = win_loss_list[0]
            master_df['t1_losses_last_20'][mask] = win_loss_list[1]
            master_df['t1_win_percent_last_20'][mask] = t1_win_percent
            master_df['t2_wins_last_20'][mask] = win_loss_list[3]
            master_df['t2_losses_last_20'][mask] = win_loss_list[4]
            master_df['t2_win_percent_last_20'][mask] = t2_win_percent    
    
    return master_df


#Input: American Odds, and Probability of a Winning Bet
#Output: Bet EV based on a $100 bet
def get_bet_ev(odds, prob):
    if odds>0:
        return ((odds * prob) - (100 * (1-prob)) )
    else:
        return ((100 / abs(odds))*100*prob - (100 * (1-prob)))


#Input: American Odds
#Output: Profit on a successful bet
def get_bet_return(odds):
    if odds>0:
        return odds
    else:
        return (100 / abs(odds))*100
    

#Input DF must have these columns:
#t1_odds (American)
#t2_odds (American)
#t1_prob (0->1)
#t2_prob (0->1)
#winner (0 or 1)
#OUTPUT: Profit per bet (based on bet of $100)

def get_ev_from_df(ev_df, print_stats = False, min_ev = 0, get_total = True):
    num_matches = 0
    num_bets = 0
    num_wins = 0
    num_losses= 0
    num_under= 0
    num_under_losses = 0
    num_under_wins = 0
    num_even = 0
    num_even_losses = 0
    num_even_wins = 0
    num_fav = 0
    num_fav_wins = 0
    num_fav_losses = 0
    profit = 0
    profit_per_bet = 0
    profit_per_match = 0    

    for index, row in ev_df.iterrows():
        num_matches = num_matches+1
        t1_bet_ev = get_bet_ev(row['t1_odds'], row['t1_prob'])
        #print(f"ODDS:{row['t1_odds']} PROB: {row['t1_prob']} EV: {t1_bet_ev}")
        t2_bet_ev = get_bet_ev(row['t2_odds'], row['t2_prob'])
        #print(f"ODDS:{row['t2_odds']} PROB: {row['t2_prob']} EV: {t2_bet_ev}")
        #print()
        
        t1_bet_return = get_bet_return(row['t1_odds'])
        t2_bet_return = get_bet_return(row['t2_odds'])
        
        
        if (t1_bet_ev > min_ev or t2_bet_ev > min_ev):
            num_bets = num_bets+1

            
        if t1_bet_ev > min_ev:
            if row['winner'] == 0:
                num_wins += 1
                profit = profit + t1_bet_return
                #print(t1_bet_return)
            elif row['winner'] == 1:
                num_losses += 1
                profit = profit - 100
            if (t1_bet_return > t2_bet_return):
                num_under += 1
                if row['winner'] == 0:
                    num_under_wins += 1
                elif row['winner'] == 1:
                    num_under_losses += 1
            elif (t1_bet_return < t2_bet_return):
                num_fav += 1
                if row['winner'] == 0:
                    num_fav_wins += 1
                elif row['winner'] == 1:
                    num_fav_losses += 1
            else:
                num_even += 1
                if row['winner'] == 0:
                    num_even_wins += 1
                elif row['winner'] == 1:
                    num_even_losses += 1

        if t2_bet_ev > min_ev:
            if row['winner'] == 1:
                num_wins += 1                    
                profit = profit + t2_bet_return
            elif row['winner'] == 0:
                num_losses += 1
                profit = profit - 100
            if (t2_bet_return > t1_bet_return):
                num_under += 1
                if row['winner'] == 1:
                    num_under_wins += 1
                elif row['winner'] == 0:
                    num_under_losses += 1
            elif (t2_bet_return < t1_bet_return):
                num_fav += 1
                if row['winner'] == 1:
                    num_fav_wins += 1
                elif row['winner'] == 0:
                    num_fav_losses += 1
            else:
                num_even += 1
                if row['winner'] == 1:
                    num_even_wins += 1
                elif row['winner'] == 0:
                    num_even_losses += 1
    if num_bets > 0:        
        profit_per_bet = profit / num_bets
    else:
        profit_per_bet = 0
    if num_matches > 0:
        profit_per_match = profit / num_matches
    else:
        profit_per_match = 0
        
    if print_stats:
        print(f"""
          Number of matches: {num_matches}
          Number of bets: {num_bets}
          Number of winning bets: {num_wins}
          Number of losing bets: {num_losses}
          Number of underdog bets: {num_under}
          Number of underdog wins: {num_under_wins}
          Number of underdog losses: {num_under_losses}
          Number of Favorite bets: {num_fav}
          Number of favorite wins: {num_fav_wins}
          Number of favorite losses: {num_fav_losses}
          Number of even bets: {num_even}
          Number of even wins: {num_even_wins}
          Number of even losses: {num_even_losses}
          Profit: {profit}
          Profit per bet: {profit_per_bet}
          Profit per match: {profit_per_match}
          
          """)
    if (get_total):
        #print(f"# Matches: {num_matches}, # Bets: {num_bets} # Wins: {num_wins}")
        return(profit)
    else:
        return (profit_per_bet)
            

#Input the train df and model and we will return a customer 5x cross validation score based off of expected value
#t1_odds and t2_odd MUST be the last 2 columns or this will break.

def custom_cv_eval(df, m, labels, odds, min_ev=0, verbose=False, get_total=True):
    X = np.array(df)
    y = np.array(labels)
    odds = np.array(odds)
    running_total = 0
    count=1
    kf = KFold(n_splits=5, shuffle=True, random_state=75)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        odds_train, odds_test = odds[train_index], odds[test_index]
        #display(y_train)
        m.fit(X_train, y_train)
        probs=m.predict_proba(X_test)
        #print(probs)
        #We need to prep the dataframe to evaluate....
        #X_odds = X_test[['t1_odds', 't2_odds']]
        #print(X_test)
        #print(X_test[:, -1])
        #print(X_test[:, -2])
        X_odds = list(zip(odds_test[:, -2], odds_test[:, -1], probs[:, 0], probs[:, 1], y_test))
        ev_prepped_df = pd.DataFrame(X_odds, columns=['t1_odds', 't2_odds', 't1_prob', 't2_prob', 'winner'])
        #display(ev_prepped_df)
        #display(temp_df)
        #print(f"{count}: {get_ev_from_df(ev_prepped_df, print_stats = False)}")
        count=count+1
        running_total = running_total + get_ev_from_df(ev_prepped_df, print_stats = verbose, min_ev = min_ev, get_total=get_total)
        #display(ev_prepped_df)
    
    return running_total




def get_best_features(features, model, df, current_features, scale=False):
    best_feature = ""
    winner_labels = df['winner_label'].copy()
    initial_df = df[current_features]
    #display(initial_df)
    #display(winner_labels)
    
    best_score = custom_cv_eval(df[current_features], model)
    best_feature = ""
    
    print(f"Current best score is: {best_score}")
    for f in features:
        if f not in current_features:
            new_features = [f] + current_features
            df_sel=df[new_features]
            if scale == True:
                sc = StandardScaler()
                df_sel = sc.fit_transform(df_sel)
            new_score = custom_cv_eval(df_sel, model)
            #print(f"Total score for {f} is: {new_score}")
            if new_score > best_score:
                best_score = new_score
                best_feature = f
            #print()
    #Keep running until we don't improve
    if best_feature != "":
        print(f"The best feature was {best_feature}.  It scored {best_score}")
        current_features = [best_feature] + current_features
        
        return(get_best_features(features, model, df, current_features, scale))
    else:
        print("NO IMPROVEMENT")
        print(f"FINAL BEST SCORE: {best_score}")
        return current_features
    

def get_best_features_v2(pos_features, m, df, cur_features, labels, odds, scale=False, min_ev=0):
    best_feature = ''
        
    #If there are no current features...
    if len(cur_features) == 0:
        best_score = -100
    else:
        df_sel = df[cur_features]
        df_sel = df_sel.dropna()
        df_sel = pd.get_dummies(df_sel)
        #OK we need to filter the labels and odds based off of the indices
        labels_sel = labels[labels.index.isin(df_sel.index)]
        odds_sel = odds[odds.index.isin(df_sel.index)]        
        best_score = custom_cv_eval(df_sel, m, labels_sel, odds_sel, min_ev=min_ev, get_total=True)
        
    best_feature = ""
    
    print(f"Current best score is: {best_score}")
    #Go thru every feature and test it...
    for f in pos_features:
        #If f is not a current feature
        if f not in cur_features:
            new_features = [f] + cur_features
            df_sel = df[new_features]
            df_sel = df_sel.dropna()
            df_sel = pd.get_dummies(df_sel)
            #display(df_sel)
            #OK we need to filter the labels and odds based off of the indices
            labels_sel = labels[labels.index.isin(df_sel.index)]
            odds_sel = odds[odds.index.isin(df_sel.index)]
            new_score = custom_cv_eval(df_sel, m, labels_sel, odds_sel, min_ev=min_ev, get_total=True)
            #print(f"{len(df_sel)} {len(labels_sel)} {len(odds_sel)}")
            if new_score > best_score:
                print(f"Feature: {f} Score: {new_score}")
                best_score = new_score
                best_feature = f
    if best_feature != "":
        print(f"The best feature was {best_feature}.  It scored {best_score}")
        cur_features = [best_feature] + cur_features
        #Keep running until we don't improve
        return(get_best_features_v2(pos_features, m, df, cur_features, labels, odds, scale, min_ev=min_ev))
    else:
        print("NO IMPROVEMENT")
        print(f"FINAL BEST SCORE: {best_score}")
        return cur_features                
                
    return []
    
    
def get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev = 0, verbose=False, get_total=True):
    df_sel = input_df[input_features]
    df_sel = df_sel.dropna()
    df_sel = pd.get_dummies(df_sel)
    labels_sel = input_labels[input_labels.index.isin(df_sel.index)]
    odds_sel = odds_input[odds_input.index.isin(df_sel.index)] 
    best_score = custom_cv_eval(df_sel, input_model, labels_sel, odds_sel, min_ev = min_ev, verbose=verbose, 
                                get_total=get_total)
    return best_score



def tune_LogisticRegression(input_model, input_features, input_df, input_labels, odds_input, min_ev=0):
    ###############################################################################################################
    #Parameters we are going to fine-tune:
    #1. penalty ('l1' or 'l2')
    #2. tol (original_value, original_value * 1.2, original_value * 0.8, rand(0, 10)
    #3. random_state = 75
    #4. solver = 'newton-cg', 'lbfgs', 'sag', 'saga'    
    ###############################################################################################################
    print()
    print()
    print("Starting New Run for LogisticRegression")
    print()
    print()
    output_model = input_model
    best_score = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=min_ev)
    print("Previous Best Score:", best_score)    

    
    penalty = ['l1', 'l2', 'none']
    solver = ['newton-cg', 'lbfgs', 'sag', 'saga']
    tol = [input_model.tol, input_model.tol * 1.2, input_model.tol * .8, random.random() * 10 ]
    for s in solver:
        score = -10000
        for p in penalty:
            for t in tol:
                if ((s == 'newton-cg') & (p == 'l1')) |\
                ((s == 'lbfgs') & (p == 'l1')) |\
                ((s == 'sag') & (p == 'l1')):

                    pass
                else:
                    test_model = LogisticRegression(solver = s, penalty = p, tol=t, random_state=75, max_iter=50000)
                    score = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=min_ev)
                    if score > best_score:
                        best_score = score
                        output_model = test_model
                        
                        print()
                        print("NEW BEST SCORE")
                        print("solver:", s, 
                              "penalty:", p,
                              "tol:", t,
                              "Best Score:", best_score)        
                        print()
                        print()
                    else:
                        pass
                        print("solver:", s, 
                              "penalty:", p,
                              "tol:", t,
                              "Score:", score)                                                       
    return(output_model)

def tune_DecisionTreeClassifier(input_model, input_features, input_df, input_labels, odds_input, min_ev=0):
    ###############################################################################################################
    #Parameters we are going to fine-tune:
    #1. criterion ('gini', 'entropy')
    #2. splitter ('random', 'best')
    #3. max_depth ('none', IF A NUMBER EXISTS +1, -1, random, else 2 RANDOM INTS 1->100)
    #4. min_samples_leaf(n-1, 0,  n+1)
    #5. max_leaf_nodes:('none', n+1, n-1, OR 4 random numbers)
    ###############################################################################################################
    print()
    print()
    print("Starting New Run for DecisionTree")
    print()
    print()
    output_model = input_model
    best_score = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=min_ev)
    print("Previous Best Score:", best_score)    

    criterion = ['gini', 'entropy']
    splitter = ['random', 'best']
    if input_model.max_depth == None:
        max_depth = [None, random.randrange(100), random.randrange(100)]
    else:
        max_depth = [input_model.max_depth, input_model.max_depth - 1, input_model.max_depth + 1, random.randrange(100)]
        max_depth = [i for i in max_depth if i > 0]

    min_samples_leaf = [input_model.min_samples_leaf, input_model.min_samples_leaf - 1,
                         input_model.min_samples_leaf + 1, random.randrange(100)]
    min_samples_leaf = [i for i in min_samples_leaf if i > 0]    
    if input_model.max_leaf_nodes == None:
        max_leaf_nodes = [None, random.randrange(1000), random.randrange(1000)]
    else:
        max_leaf_nodes = [input_model.max_leaf_nodes, input_model.max_leaf_nodes - 1, 
                     input_model.max_leaf_nodes + 1, random.randrange(1000)]
        max_leaf_nodes = [i for i in max_leaf_nodes if i > 0]
    
    for l in max_leaf_nodes:
        for sam in min_samples_leaf:
            for m in max_depth:
                for c in criterion:
                    for s in splitter:
                        test_model = DecisionTreeClassifier(criterion = c, splitter = s, max_depth = m,
                                                            min_samples_leaf=sam, max_leaf_nodes = l, random_state=75)
                        score = get_ev(input_df, test_model, input_features, input_labels, odds_input, min_ev=min_ev)
                        if score > best_score:
                            best_score = score
                            output_model = test_model
                            print()
                            print("NEW BEST SCORE")
                            
                            print("Criterion:", c, "splitter:", s, "max_depth:", m, 
                                  "min_samples_leaf:", sam, "max_leaf_nodes:", l, best_score)        
                            print()
                        else:
                            pass
                            print("Criterion:", c, "splitter:", s, "max_depth:", m, 
                                  "min_samples_leaf:", sam, "max_leaf_nodes:", l, score)        
                            
                                        
    
    return output_model

def tune_RandomForestClassifier(input_model, input_features, input_df, input_labels, odds_input, min_ev=0):
    ###############################################################################################################
    #Parameters we are going to fine-tune:
    #1. criterion ('gini', 'entropy')
    #2. max_features ('auto', 'sqrt', 'log2')
    #3. max_depth ('none', IF A NUMBER EXISTS +2, -2, ELSE 2 RANDOM INTS 1->100)
    #4. min_samples_leaf(n-2, 0, n+2)
    #5. max_leaf_nodes:('none', n+2, n-2, OR 2 random numbers)
    #6. n_estimators: (n, n+2, n-2)
    ###############################################################################################################    
    print()
    print()
    print("Starting New Run for RandomForestClassifier")
    print()
    print()
    output_model = input_model
    best_score = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=min_ev)
    print("Previous Best Score:", best_score)        
    #1. criterion ('gini', 'entropy')
    criterion = ['gini', 'entropy']
    #2. max_features ('auto', 'log2')
    max_features = ['auto', 'log2', None]
    #3. max_depth ('none', IF A NUMBER EXISTS +2, +4, -2, -4 ELSE 4 RANDOM INTS 1->100)
    if input_model.max_depth == None:
        max_depth = [None, random.randrange(100), random.randrange(100)]
    else:
        max_depth = [input_model.max_depth, input_model.max_depth - 2,   
                     input_model.max_depth + 2, random.randrange(100)]
        max_depth = [i for i in max_depth if i > 0]
    #4. min_samples_leaf(n-1, n-2, 0,  n+1, n+2)
    min_samples_leaf = [input_model.min_samples_leaf, input_model.min_samples_leaf - 2, 
                         input_model.min_samples_leaf + 2, random.randrange(100)]
    min_samples_leaf = [i for i in min_samples_leaf if i > 0]
    
    #5. max_leaf_nodes:('none', n+1, n+2, n-1, n-2, OR 4 random numbers)
    if input_model.max_leaf_nodes == None:
        max_leaf_nodes = [None, random.randrange(1000), random.randrange(1000)]
    else:
        max_leaf_nodes = [input_model.max_leaf_nodes, input_model.max_leaf_nodes - 2,  
                     input_model.max_leaf_nodes + 2, random.randrange(1000)]
        max_leaf_nodes = [i for i in max_leaf_nodes if i > 0]
    n_estimators = [input_model.n_estimators, input_model.n_estimators - 2,   
                 input_model.n_estimators + 2, random.randrange(200)]
    n_estimators = [i for i in n_estimators if i > 0]
    
    
    
    for n in n_estimators:
        for ml in max_leaf_nodes:
            for ms in min_samples_leaf:
                for md in max_depth:
                    for mf in max_features:
                        for c in criterion:
                            test_model = RandomForestClassifier(n_estimators = n, max_leaf_nodes = ml, 
                                                                min_samples_leaf = ms,
                                                                max_depth = md, criterion = c, 
                                                                max_features = mf, 
                                                                n_jobs = -1,
                                                                random_state=75)
                            score = get_ev(input_df, test_model, input_features, input_labels, odds_input, min_ev=min_ev)
                            if score > best_score:
                                best_score = score
                                output_model = test_model
                                print()
                                print("NEW BEST SCORE")
                                print("Criterion:", c, "max_features:", mf, "max_depth:", md, "min_samples_leaf:", ms,
                                      "max_leaf_nodes:", ml, "n_estimators", n, best_score)        
                                print()
                                print()
                            else:
                                pass
                                print("Criterion:", c, "max_features:", mf, "max_depth:", md, "min_samples_leaf:", ms,
                                      "max_leaf_nodes:", ml, "n_estimators", n, score)        
                            
    return output_model

def tune_GradientBoostingClassifier(input_model, input_features, input_df, input_labels, odds_input, min_ev=0):
    ###############################################################################################################
    #Parameters we are going to fine-tune:
    #1. criterion ('friedman_mse', 'mse', 'mae')
    #2. loss ('deviance', 'exponential')
    #3. n_estimators (n, n+1, n-1)
    #4. learning_rate (learning_rate, learning_rate *1.1, learning_rate*.9)
    #5. min_samples_leaf: (n, n-1, n+1)
    #6. max_depth: (n, n+1, n-1)
    #7. max_features: (None, 'auto', 'sqrt', 'log2')
    #8. max_leaf_nodes: (None, n+1, n-1, OR 2 random numbers)
    #9. tol (n, n*1.1, n*.9)
    ###############################################################################################################  
    print()
    print()
    print("Starting New Run")
    print()
    print()
    output_model = input_model
    best_score = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=min_ev)
    print("Previous Best Score:", best_score)
    
    #1. criterion ('friedman_mse', 'mse', 'mae')
    criterion = ['friedman_mse']
    
    #2. loss ('deviance', 'exponential')
    loss = ['deviance']

    #3. n_estimators (n, n+1, n-1)
    n_estimators = [input_model.n_estimators, input_model.n_estimators - 1,  input_model.n_estimators + 1,
                    random.randrange(200)]
    n_estimators = [i for i in n_estimators if i > 0]    
    
    #4. learning_rate (learning_rate, learning_rate *1.1, learning_rate*.9)
    learning_rate = [input_model.learning_rate]
    
    #5. min_samples_leaf: (n, n-1, n+1)
    min_samples_leaf = [input_model.min_samples_leaf, input_model.min_samples_leaf - 1,
                         input_model.min_samples_leaf + 1]
    min_samples_leaf = [i for i in min_samples_leaf if i > 0]

    #6. max_depth: (n, n+1, n-1)
    if input_model.max_depth == None:
        max_depth = [None, random.randrange(100), random.randrange(100)]
    else:
        max_depth = [input_model.max_depth, input_model.max_depth - 1,  
                     input_model.max_depth + 1, random.randrange(100)]
        max_depth = [i for i in max_depth if i > 0]
        
    #7. max_features: (None, 'auto', 'sqrt', 'log2')
    max_features = ['sqrt', 'log2', None]

    #8. max_leaf_nodes: (None, n+1, n-1, OR 2 random numbers)
    if input_model.max_leaf_nodes == None:
        max_leaf_nodes = [None, random.randrange(1000), random.randrange(1000)]
    else:
        max_leaf_nodes = [input_model.max_leaf_nodes, input_model.max_leaf_nodes - 1, input_model.max_leaf_nodes + 1, 
                          random.randrange(1000)]
        max_leaf_nodes = [i for i in max_leaf_nodes if i > 0]

    #9. tol (n, n*1.1, n*.9)
    tol = [input_model.tol, input_model.tol * 1.2, input_model.tol * .8]
            
    print(len(tol) * len(max_leaf_nodes) * len(max_features) * len(max_depth) * len(min_samples_leaf) * len(learning_rate) * len(n_estimators) * len(loss) * len(criterion))    
        
        
    for t in tol:
        for ml in max_leaf_nodes:    
            for mf in max_features:
                for md in max_depth:
                    for ms in min_samples_leaf:
                        for lr in learning_rate:
                            for n in n_estimators:
                                for l in loss:
                                    for c in criterion:
                                        test_model = GradientBoostingClassifier(n_estimators = n, 
                                                                                learning_rate = lr,
                                                                                criterion = c,
                                                                                min_samples_leaf = ms,
                                                                                max_depth = md,
                                                                                loss = l, 
                                                                                max_features = mf,
                                                                                max_leaf_nodes = ml,
                                                                                tol = t,
                                                                                random_state=75)
                                        score = get_ev(input_df, test_model, input_features, input_labels, odds_input, min_ev=min_ev)
                                        if score > best_score:
                                            best_score = score
                                            output_model = test_model
                                            print()
                                            print("NEW BEST SCORE")
                                            print("Criterion:", c,
                                                  "n_estimators:", n,
                                                  "Loss:", l,
                                                  "Learning Rate:", lr,
                                                  "Min Samples/Leaf:", ms,
                                                  "Max Depth:", md,
                                                  "Max Features:", mf,
                                                  "Max Leaf Nodes:", ml,
                                                  "tol:", t,
                                                  "Best Score:", best_score)        
                                            print()
                                            print()
                                        else:
                                            pass
                                            print("Criterion:", c,
                                                  "n_estimators:", n,                          
                                                  "Loss:", l, 
                                                  "Learning Rate:", lr,
                                                  "Min Samples/Leaf:", ms,
                                                  "Max Depth:", md,
                                                  "Max Features:", mf,
                                                  "Max Leaf Nodes:", ml,
                                                  "tol:", t,
                                                  "Score:", score)        

    
    return(output_model)

def tune_GaussianNB(input_model, input_features, input_df, input_labels, odds_input, min_ev=0):
    ###############################################################################################################
    #Parameters we are going to fine-tune:
    #1. var_smoothing (1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6)
    ###############################################################################################################  
    print()
    print()
    print("Starting New Run for GaussianNB")
    print()
    print()
    output_model = input_model
    best_score = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=min_ev)
    print("Previous Best Score:", best_score)    
    
    var_smoothing = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    
    for v in var_smoothing:
        test_model = GaussianNB(var_smoothing = v)
        score = get_ev(input_df, test_model, input_features, input_labels, odds_input, min_ev=min_ev)
        if score > best_score:
            best_score = score
            output_model = test_model
            print()
            print("NEW BEST SCORE")
            print("var_smoothing:", v, 
                  "Best Score:", best_score)        
            print()
            print()
        else:
            pass
            print("var_smoothing:", v, 
                  "Score:", score)        
        
    
    return output_model

def tune_hyperparameters(input_model, input_features, input_df, input_labels, odds_input, min_ev=0):
    best_model = input_model
    keep_going = True
    
    if isinstance(input_model, LogisticRegression):
        while(keep_going):
            pos_model = (tune_LogisticRegression(best_model, input_features, input_df, input_labels, odds_input, min_ev=min_ev))
            
            if str(pos_model) == str(best_model):  #Direct comparisons don't seem to work....
                keep_going = False
                output_model = best_model
            else:
                best_model = pos_model
                
    elif isinstance(input_model, DecisionTreeClassifier):
        while(keep_going):
            pos_model = (tune_DecisionTreeClassifier(best_model, input_features, input_df, input_labels, odds_input, min_ev=min_ev))
            if str(pos_model) == str(best_model):  #Direct comparisons don't seem to work....
                keep_going = False
                output_model = best_model
            else:
                best_model = pos_model            
                
    elif isinstance(input_model, RandomForestClassifier):
        while(keep_going):
            pos_model = (tune_RandomForestClassifier(best_model, input_features, input_df, input_labels, odds_input, min_ev=min_ev))
            if str(pos_model) == str(best_model):  #Direct comparisons don't seem to work....
                keep_going = False
                output_model = best_model
            else:
                best_model = pos_model    
                                
    elif isinstance(input_model, GradientBoostingClassifier):
        print("HI")
        while(keep_going):
            pos_model = (tune_GradientBoostingClassifier(best_model, input_features, input_df, input_labels, odds_input, min_ev=min_ev))
            if str(pos_model) == str(best_model):  #Direct comparisons don't seem to work....
                keep_going = False
                output_model = best_model
            else:
                best_model = pos_model                    
                
    elif isinstance(input_model, GaussianNB):
        while(keep_going):
            pos_model = (tune_GaussianNB(best_model, input_features, input_df, input_labels, odds_input, min_ev=min_ev))
            if str(pos_model) == str(best_model):  #Direct comparisons don't seem to work....
                keep_going = False
                output_model = best_model
            else:
                best_model = pos_model                    
                
                
    else:
        output_model = input_model
    return(output_model)                

def tune_ev(input_model, input_features, input_df, input_labels, odds_input, verbose=False):
    best_ev = 0
    best_pos = -1
    for temp_ev in range(200):
        pos_ev = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=temp_ev, verbose=verbose,
                       get_total=True)
        print(temp_ev, pos_ev)
        if pos_ev > best_ev:
            best_ev = pos_ev
            best_pos = temp_ev
    return best_pos
    
    
def remove_to_improve(cur_features, m, df, labels, odds, scale=False, min_ev = 0):
    #If the list is empty we can just return it without doing anything
    number_of_features = len(cur_features)
    df_sel = df[cur_features]
    df_sel = df_sel.dropna()
    df_sel = pd.get_dummies(df_sel)
    labels_sel = labels[labels.index.isin(df_sel.index)]
    odds_sel = odds[odds.index.isin(df_sel.index)]        
    orig_score = custom_cv_eval(df_sel, m, labels_sel, odds_sel, get_total=True, min_ev = min_ev)
    #print(orig_score)
    best_features = cur_features
    best_score = orig_score
    print(f"The original score is {orig_score}")
    if number_of_features == 0:
        return []
    
    for z in range(number_of_features):
        temp_features = cur_features.copy()
        #Remove a feature
        del temp_features[z]
        df_sel = df[temp_features]
        df_sel = df_sel.dropna()
        df_sel = pd.get_dummies(df_sel)
        labels_sel = labels[labels.index.isin(df_sel.index)]
        odds_sel = odds[odds.index.isin(df_sel.index)]        
        temp_score = custom_cv_eval(df_sel, m, labels_sel, odds_sel, get_total=True, min_ev = min_ev)
        if temp_score > best_score:
            best_features = temp_features
            best_score = temp_score
            print(f"NEW BEST FEATURE SET")
            print(best_features)
            print(best_score)
        else:
            print("Score: ", temp_score)
        
        #Get a score
    if best_features != cur_features:
        return remove_to_improve(best_features, m, df, labels, odds, scale, min_ev)
    else:
        return best_features    
    
    
def evaluate_model(input_model, input_features, input_ev, train_df, train_labels, train_odds, test_df, test_labels,
                  test_odds, verbose=True):
    model_score = 0
    
    df_train = train_df[input_features].copy()
    df_test = test_df[input_features].copy()
    df_train = df_train.dropna()
    df_test = df_test.dropna()
        
    df_train = pd.get_dummies(df_train)
    df_test = pd.get_dummies(df_test)
    df_train, df_test = df_train.align(df_test, join='left', axis=1)    #Ensures both sets are dummified the same
    df_test = df_test.fillna(0)

    #LOOK AT get_ev and prepare the labels and odds
    
    labels_train = train_labels[train_labels.index.isin(df_train.index)]
    odds_train = train_odds[train_odds.index.isin(df_train.index)] 
    labels_test = test_labels[test_labels.index.isin(df_test.index)]
    odds_test = test_odds[test_odds.index.isin(df_test.index)] 
    
    
    
    display(df_train.shape)
    display(labels_train.shape)
    display(odds_train.shape)
    display(df_test.shape)
    display(labels_test.shape)
    display(odds_test.shape)
    
    input_model.fit(df_train, labels_train)

    
    
    probs = input_model.predict_proba(df_test)

    
    odds_test = np.array(odds_test)    
    
    
    prepped_test = list(zip(odds_test[:, -2], odds_test[:, -1], probs[:, 0], probs[:, 1], labels_test))
    ev_prepped_df = pd.DataFrame(prepped_test, columns=['t1_odds', 't2_odds', 't1_prob', 't2_prob', 'winner'])
    
    display(ev_prepped_df)
    
    #display(df_test)
    #display(df_test)
    model_score = get_ev_from_df(ev_prepped_df, print_stats = True, min_ev = input_ev, get_total=True)
    
    return(model_score)    