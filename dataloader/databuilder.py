# Define several utility functions
def list_of_sites(x):
  sites = []
  if len(x) != 0:
      sites = [item.keys() for item in x]
  else:
      sites = ['NA']

  return sites
  
  
def build_table(df, site_name, exit_code):
  sparse_df = pd.DataFrame(columns=site_name,
                           index=exit_code).fillna(value=0).sort_index()

  if len(df.keys()) == 0 or len(df.values()) == 0:
    return sparse_df
    
  else:
    for exit_code, site_dict in zip(df.keys(), df.values()):
      #print exit_code, site_dict
      for site, count in site_dict.items():
        sparse_df.loc[exit_code, site] = 0 if math.isnan(count) else count
    
    return sparse_df


def build_table_flatten(x):
  d_outer = []
    
  for column in x: # 60 columns (i.e. sites)
      #d_outer.append([item for item in x[column]]) # 43 items
      for item in x[column]:
          d_outer.append(item)
    
  return d_outer


def combine_features(x, feature1, feature2):
  return x[feature1]+x[feature2]


def onehot(labels):
  Uniques, Index  = np.unique(labels,return_inverse=True)
  # Convert labels to categorical one-hot encoding
  # convert integers to dummy variables (i.e. one hot encoded)
  one_hot_labels = np_utils.to_categorical(Index,len(Uniques))
  return one_hot_labels


def inverse_onehot(matrix):
  labels =[]
  for row in matrix:
      labels.append(np.argmax(row,axis=0))
  return labels  


def xrootd_fnc(x, column):
  # if isinstance(x.keys(), dict): 
  if column in x.keys():
      return str(x[column])
    
  else:
      return str('NaN')


def splitting_fnc(x, column):
  if column in x.keys():
      return str(x[column])
        
  else:
      return str('1x')


def merge_labels(x, features):
  merged_label = '_'.join(x[features]) 

  return merged_label


def plot_class_count(x, feature):
  
  data = x[[feature]]
  print(data.groupby(feature).size())
  w, h = 15, 7
  plt.figure(figsize=(15,7))
  plt.tight_layout()
  
  ax = sns.countplot(x=feature, data=data)
  ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
  ax.patch.set_facecolor('#FFFFFF')
  
  return plt.show()


def exit_code_counts(x, feature, title='good'):
  # transverse the workflow table (from exit code vs site to site vs exit code)
  df = x[feature].T.sum().T.sum()
  df_histo = df.to_dict()
  df_histo = collections.OrderedDict(sorted(df_histo.items()))

  ax = df.plot(kind='bar',
               stacked=True,
               figsize=(10,6),
               align='center')

  for container in ax.containers:
    plt.setp(container, width=1)

  width = 1.0

  pos = np.arange(len(df_histo.keys()))
  width = 1.     # gives histogram aspect to the bar diagram

  ax = plt.axes()
  ax.set_xticks(pos + (width / 2))
  ax.set_xticklabels(df_histo.keys())

  ax.patch.set_facecolor('#FFFFFF')
  x0, x1 = ax.get_xlim()
  ax.set_xlim(x0 -0.5, x1 + 0.25)
  ax.set_xlabel(title+"_site_exit_code")
  ax.set_ylabel("Count_exit_code")
  patches, labels = ax.get_legend_handles_labels()
  ax.legend(patches, labels, loc='best')
  plt.xticks(rotation='vertical')
  plt.tight_layout()
  
  return plt.show()


def extract_campaign(x):

    campaign = re.match(r'[^_]*_(task_)?.*?([^-_]*)(-\d*)?_.*', x['index']).group(2)

    return campaign


def plot_campaign_exit_codes(campaign_name = "RunIISummer17DRPremix", 
                             exit_code_feature='errors_good_sites_exit_codes'):

  # flatten list of tuples containing exit codes
  lst = data_index_reset.query('campaign == @campaign_name')[exit_code_feature]
  lst = [list(x) for x in lst]

  result= list(itertools.chain(*lst))
  
  campaign_exit_code_counts = Counter(elem for elem in result)
  campaign_exit_code_counts = collections.OrderedDict(sorted(campaign_exit_code_counts.items(), 
                                                             key=lambda s: s[0]))
  
  # gives histogram aspect to the bar diagram
  width = 1.0     
  
  #Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, 
  #stab10, tab10_r, tab20, tab20_r, tab20b, 

  # plot exit codes occurrences
  ax = pd.DataFrame.from_dict(campaign_exit_code_counts, 
                              orient='index').plot(kind='bar',
                                                   colormap=plt.cm.get_cmap('tab20'),
                                                   #stacked=True,
                                                   figsize=(10,6),
                                                   align='center')
  
  for container in ax.containers:
    plt.setp(container, width=1)
    
  pos = np.arange(len(campaign_exit_code_counts.keys()))

  ax.set_xticks(pos + (width / 2))

  ax.patch.set_facecolor('#FFFFFF')
  ax.set_xlabel("Site_exit_code")
  ax.set_ylabel("Exit_code_count")
  patches, labels = ax.get_legend_handles_labels()

  
  patch = mpatches.Patch(color='#3774B1',#'tab:blue', 
                         label=str(campaign_name))
  plt.title('Exit code histogram of campaign: '+str(campaign))
  plt.legend(handles=[patch], title="campaign",
             fontsize='small', fancybox=True)
  
  ax.set_xticklabels(campaign_exit_code_counts.keys(), rotation=45)
  plt.tight_layout()
  plt.savefig("datalab/"+str(campaign_name)+'_'+str(exit_code_feature)+'.png', 
              bbox_inches='tight')
  
  return plt.show()
