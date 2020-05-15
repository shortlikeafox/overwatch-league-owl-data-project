import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

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

def get_ev_from_df(ev_df, print_stats = False, min_ev = 0):
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
            
    profit_per_bet = profit / num_bets
    profit_per_match = profit / num_matches
        
        
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
        
    return (profit_per_bet)
            

#Input the train df and model and we will return a customer 5x cross validation score based off of expected value
#t1_odds and t2_odd MUST be the last 2 columns or this will break.

def custom_cv_eval(df, m, min_ev = 0):
    #We need to split away the winner...
    y = df['winner_label'].copy()
    #display(y)
    X = df.drop('winner_label', axis=1)
    
    ##We need a numpy array it seems like...
    X = np.array(X)
    y = np.array(y)
    #display(X)
    running_total = 0
    count=1
    kf = KFold(n_splits=5, shuffle=True, random_state=75)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #print(test_index)
        m.fit(X_train, y_train)
        probs=m.predict_proba(X_test)
        #We need to prep the dataframe to evaluate....
        #X_odds = X_test[['t1_odds', 't2_odds']]
        #print(X_test)
        #print(X_test[:, -1])
        #print(X_test[:, -2])
        X_odds = list(zip(X_test[:, -2], X_test[:, -1], probs[:, 0], probs[:, 1], y_test))
        ev_prepped_df = pd.DataFrame(X_odds, columns=['t1_odds', 't2_odds', 't1_prob', 't2_prob', 'winner'])
        #display(temp_df)
        #print(f"{count}: {get_ev_from_df(ev_prepped_df, print_stats = False)}")
        count=count+1
        running_total = running_total + get_ev_from_df(ev_prepped_df, print_stats = False, min_ev = min_ev)
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